# visualizer.py
"""
Lightweight visualization facade with a pluggable backend.

Goals:
- One-time init(), then visualizer.log(...)
- Backend chosen by env (VISUALIZER=rerun|disabled), default = rerun if available
- Safe to import anywhere; does nothing if disabled or unavailable
- Works across multi-node/multi-process jobs (rank-aware namespacing)
- Designed to later support wandb, gradio, etc. via the backend registry

Current support: rerun.io only (optional dependency)
"""

import atexit
import importlib.util  # Add this import
import logging
import os
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import numpy as np
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers

# Optional imports (gate behind backend)
_HAS_RERUN = importlib.util.find_spec("rerun") is not None
_HAS_WANDB = importlib.util.find_spec("wandb") is not None

logger = logging.getLogger(__name__)
_VISUALIZER_LOG_PREFIX = "[Visualizer] "

# Optional Drake import for RigidTransform convenience
try:
    from pydrake.math import RigidTransform  # type: ignore

    _HAS_DRAKE = True
except Exception:
    _HAS_DRAKE = False

# ---------------------------
# Backend interface + registry
# ---------------------------


class Backend:
    """Abstract backend API. Keep this minimal."""

    name: str = "abstract"

    def init(self, run_name: str, **kwargs) -> None:  # noqa: D401
        """Initialize the session/run."""
        raise NotImplementedError

    def log(self, path: str, value: Any, **kwargs) -> None:
        """Log a value at a hierarchical path."""
        raise NotImplementedError

    def flush(self) -> None:
        pass  # Optional

    def shutdown(self) -> None:
        pass  # Optional


_BACKENDS: dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    _BACKENDS[backend.name] = backend


# ---------------------------
# Global state / facade
# ---------------------------


@dataclass
class _State:
    backend_name: str = "auto"  # "rerun" | "disabled" | ...
    backend: Backend | None = None
    run_name: str = ""
    enabled: bool = False
    initialized: bool = False
    rank_prefix: str = ""  # used to namespace multi-process logs

    # Tracks if user explicitly called disable() before init().
    # If True, init() will respect that and not auto-enable.
    user_disabled: bool = False

    # Per-(log method, path) counters used for sparse logging (every-n subsampling).
    # Example key: ("images", "observation_images")
    counters: dict[tuple[str, str], int] = field(default_factory=dict)


_STATE = _State()


def _detect_rank_prefix() -> str:
    # Common envs: torchrun/SLURM/MPI. Keep it simple.
    rank = (
        os.environ.get("RANK")
        or os.environ.get("SLURM_PROCID")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("OMPI_COMM_WORLD_RANK")
    )
    if rank is None:
        return ""
    return f"r{rank}"


def _choose_backend_from_env() -> str:
    # VISUALIZER values:
    #   disabled|off|0 -> disabled
    #   rerun | wandb (default to disabled if no env variable is set)
    val = (os.environ.get("VISUALIZER") or "").strip().lower()
    if not val:
        return "disabled"  # Default to disabled if no VISUALIZER is set
    if val in {"disabled", "off", "0", "none"}:
        return "disabled"
    if val in {"rerun", "wandb"}:
        return val
    # Auto
    return "disabled"


def _get_backend(name: str) -> Backend | None:
    if name == "disabled":
        return None
    if name == "rerun":
        if not _HAS_RERUN:
            logger.warning(f"{_VISUALIZER_LOG_PREFIX} Rerun package not available; using disabled.")
            return None
        try:
            from vla_foundry.visualizers.rerun_backend import (
                RerunBackend,
            )  # Import only when needed

            register_backend(RerunBackend())
        except ImportError as e:
            logger.warning(f"{_VISUALIZER_LOG_PREFIX} Rerun backend import failed: {e}")
            return None
    if name == "wandb":
        if not _HAS_WANDB:
            logger.warning(f"{_VISUALIZER_LOG_PREFIX} WandB package not available; using disabled.")
            return None
        try:
            from vla_foundry.visualizers.wandb_backend import (
                WandbBackend,
            )  # Import only when needed

            register_backend(WandbBackend())
        except ImportError as e:
            logger.warning(f"{_VISUALIZER_LOG_PREFIX} WandB backend import failed: {e}")
            return None
    b = _BACKENDS.get(name)
    if b is None:
        logger.warning(f"{_VISUALIZER_LOG_PREFIX} Backend '{name}' not registered; using disabled.")
    return b


def init(
    run_name: str | None = None,
    *,
    backend: str | None = None,
    spawn: bool = True,
    open_browser: bool = False,
    allow_disabled: bool = True,
    add_rank_to_run: bool = False,
) -> None:
    """
    Initialize the visualizer once per process.

    - backend: If None, automatically detect the backend using the VISUALIZER environment variable.
    - VISUALIZER=disabled disables everything.
    - run_name: Default from VISUALIZER_RUN_NAME or basename of CWD.
    - open_browser: Whether to automatically open the browser with the web viewer (rerun backend).
    - add_rank_to_run: Append "-r{rank}" to run name.
    """
    if _STATE.initialized:
        return

    # New session/run: reset sparse-logging counters so frequencies start from 0.
    _STATE.counters.clear()

    # Automatically detect the backend if not explicitly provided
    chosen = backend or _choose_backend_from_env()
    be = _get_backend(chosen)

    # Derive run name
    default_run = os.environ.get("VISUALIZER_RUN_NAME")
    if not default_run:
        default_run = os.path.basename(os.getcwd())
    rn = run_name or default_run or "run"

    rp = _detect_rank_prefix()
    if add_rank_to_run and rp:
        rn = f"{rn}-{rp}"

    _STATE.backend_name = chosen
    _STATE.backend = be
    _STATE.run_name = rn
    _STATE.rank_prefix = rp

    if be is None:
        _STATE.enabled = allow_disabled is False  # typically False
        _STATE.initialized = True
        logger.info(f"{_VISUALIZER_LOG_PREFIX} Visualizer disabled (no backend).")
        return

    # Backend init
    be.init(rn, spawn=spawn, open_browser=open_browser)
    # Respect pre-init disable() calls: only enable if user hasn't explicitly disabled.
    _STATE.enabled = not _STATE.user_disabled
    _STATE.initialized = True

    # Ensure we shut down cleanly (but not in Colab/notebook mode where spawn=False)
    if spawn:
        atexit.register(shutdown)


def enabled() -> bool:
    return _STATE.enabled and _STATE.backend is not None


def set_logging_enabled(is_enabled: bool) -> None:
    """Enable/disable logging *without* shutting down the backend.

    This is useful for evaluation loops where you want to selectively log only
    some samples/steps while keeping the same backend run/session alive.

    Can be called before init() - the setting will persist.
    """
    _STATE.enabled = bool(is_enabled)
    # Track user intent so init() respects pre-init disable() calls.
    _STATE.user_disabled = not is_enabled


def enable_logging() -> None:
    """Convenience wrapper for ``set_logging_enabled(True)``."""

    set_logging_enabled(True)


def enable() -> None:
    """Alias for :func:`enable_logging`."""

    enable_logging()


def disable_logging() -> None:
    """Convenience wrapper for ``set_logging_enabled(False)``."""

    set_logging_enabled(False)


def disable() -> None:
    """Alias for :func:`disable_logging`."""

    disable_logging()


def reset_sparse_counters() -> None:
    """Clear per-path counters used by sparse (every-n) logging."""

    _STATE.counters.clear()


def _pop_every_n(kwargs: dict[str, Any]) -> int | None:
    """Extract the sparse-logging frequency from kwargs.

    We support both ``n=...`` (as requested) and ``every_n=...``.
    The value is removed from ``kwargs`` so backends don't need to know about it.
    """

    every_n = None
    if "every_n" in kwargs:
        every_n = kwargs.pop("every_n")
    elif "n" in kwargs:
        every_n = kwargs.pop("n")

    if every_n is None:
        return None
    # Guard against bools (since bool is a subclass of int)
    if isinstance(every_n, bool):
        raise TypeError("n/every_n must be an integer >= 1")
    try:
        every_n_int = int(every_n)
    except (TypeError, ValueError) as err:
        raise TypeError("n/every_n must be an integer >= 1") from err
    if every_n_int < 1:
        raise ValueError("n/every_n must be an integer >= 1")
    return every_n_int


def _should_log(kind: str, path: str, every_n: int | None) -> bool:
    """Return True if we should log this (kind, path) event."""

    if every_n is None or every_n <= 1:
        return True

    key = (kind, path)
    count = _STATE.counters.get(key, 0) + 1
    _STATE.counters[key] = count

    # Log on the first call, then on every Nth call thereafter (N, 2N, 3N, ...)
    return (count == 1) or (count % every_n == 0)


def _prefix(path: str) -> str:
    # Namespace logs by rank to support multi-node/multi-process debugging
    if _STATE.rank_prefix:
        return f"{_STATE.rank_prefix}/{path}"
    return path


def ensure_initialized_and_enabled(func):
    """Decorator to ensure the visualizer is initialized and enabled before logging."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _STATE.initialized:
            init()
        if not enabled():
            return  # Bypass if disabled
        return func(*args, **kwargs)

    return wrapper


class Visualizer:
    """Base visualizer facade with general-purpose logging methods."""

    @ensure_initialized_and_enabled
    def log_images(self, path: str, images: Any, **kwargs) -> None:
        """Log images to the active backend.

        Args:
            path: Base path in the visualization hierarchy.
            images: Either a single NumPy array representing an image or a dictionary of images.
        """
        every_n = _pop_every_n(kwargs)
        full_path = _prefix(path)
        if not _should_log("images", full_path, every_n):
            return
        _STATE.backend.log_images(full_path, images, **kwargs)  # type: ignore[union-attr]

    @ensure_initialized_and_enabled
    def log_scalar(self, path: str, value: float, **kwargs) -> None:
        """Log a scalar value to the active backend.

        Args:
            path: Path in the visualization hierarchy (e.g., "metrics/loss").
            value: Scalar value to log.
        """
        every_n = _pop_every_n(kwargs)
        full_path = _prefix(path)
        if not _should_log("scalar", full_path, every_n):
            return
        _STATE.backend.log_scalar(full_path, value, **kwargs)  # type: ignore[union-attr]

    @ensure_initialized_and_enabled
    def log_points3d(self, path: str, points: np.ndarray, **kwargs) -> None:
        """Log 3D points to the active backend.

        Args:
            path: Path in the visualization hierarchy (e.g., "points/scene").
            points: 3D points as a NumPy array of shape (N, 3).
        """
        every_n = _pop_every_n(kwargs)
        full_path = _prefix(path)
        if not _should_log("points3d", full_path, every_n):
            return
        _STATE.backend.log_points3d(full_path, points, **kwargs)  # type: ignore[union-attr]

    @ensure_initialized_and_enabled
    def log_trajectory(self, path: str, trajectory_points: np.ndarray, **kwargs) -> None:
        """Log a trajectory as waypoints and a path in the visualization hierarchy.

        Args:
            path: Base path in the visualization hierarchy (e.g., "robot/trajectory").
            trajectory_points: Array of shape (N, 3) representing the trajectory points.
        """
        every_n = _pop_every_n(kwargs)
        full_path = _prefix(path)
        if not _should_log("trajectory", full_path, every_n):
            return

        if trajectory_points.ndim != 2 or trajectory_points.shape[1] != 3:
            raise ValueError("trajectory_points must be a (N, 3) array")
        self.log_points3d(f"{path}/waypoints", trajectory_points, **kwargs)
        self.log_line_strips3d(f"{path}/path", trajectory_points, **kwargs)

    @ensure_initialized_and_enabled
    def log_line_strips3d(self, path: str, line_strips: np.ndarray | dict[str, np.ndarray], **kwargs) -> None:
        """Log 3D line strips to the active backend.

        Args:
            path: Path in the visualization hierarchy (e.g., "lines/trajectory").
            line_strips: Line strips as a NumPy array of shape (N, 3),
                or multiple named line strips as a dictionary.
        """
        every_n = _pop_every_n(kwargs)
        full_path = _prefix(path)
        if not _should_log("line_strips3d", full_path, every_n):
            return
        _STATE.backend.log_line_strips3d(full_path, line_strips, **kwargs)  # type: ignore[union-attr]

    @ensure_initialized_and_enabled
    def log_text(self, path: str, text: str, **kwargs) -> None:
        """Log a text value to the active backend.

        Args:
            path: Path in the visualization hierarchy (e.g., "language/instruction").
            text: Text value to log.
        """
        every_n = _pop_every_n(kwargs)
        full_path = _prefix(path)
        if not _should_log("text", full_path, every_n):
            return
        _STATE.backend.log_text(full_path, text, **kwargs)  # type: ignore[union-attr]

    @ensure_initialized_and_enabled
    def log_pose(self, path: str, translation: np.ndarray, rotation: np.ndarray, **kwargs) -> None:
        """Log a generic pose to the active backend.

        Args:
            path: Path in the visualization hierarchy (e.g., "poses/end_effector").
            translation: Translation vector of shape (3,).
            rotation: Rotation representation. Can be:
                - Quaternion [x, y, z, w] of shape (4,)
                - Rotation matrix of shape (3, 3)
                - Transformation matrix of shape (4, 4) (translation will be ignored)
        """
        every_n = _pop_every_n(kwargs)
        full_path = _prefix(path)
        if not _should_log("pose", full_path, every_n):
            return

        # Ensure inputs are numpy arrays
        translation = np.asarray(translation)
        rotation = np.asarray(rotation)

        # Handle translation validation
        if translation.shape != (3,):
            raise ValueError(f"Translation must be shape (3,), got {translation.shape}")

        # Handle different rotation formats and convert to quaternion [x, y, z, w]
        if rotation.shape == (4,):
            # Already quaternion [x, y, z, w] - validate it's normalized
            quaternion = rotation.copy()
            quat_norm = np.linalg.norm(quaternion)
            if quat_norm < 1e-6:
                raise ValueError("Zero-norm quaternion is invalid")
            if not np.isclose(quat_norm, 1.0, atol=1e-3):
                # Normalize quaternion if it's not already normalized
                quaternion = quaternion / quat_norm
            final_translation = translation
        elif rotation.shape == (3, 3):
            # Rotation matrix - validate and convert to quaternion
            try:
                from scipy.spatial.transform import Rotation

                # Validate that it's a proper rotation matrix
                if not np.allclose(np.dot(rotation, rotation.T), np.eye(3), atol=1e-6):
                    raise ValueError("Matrix is not orthogonal (not a valid rotation matrix)")
                if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
                    raise ValueError("Matrix determinant is not 1 (not a proper rotation matrix)")
                quaternion = Rotation.from_matrix(rotation).as_quat()  # Returns [x, y, z, w]
                final_translation = translation
            except ImportError as err:
                raise ImportError("scipy is required for rotation matrix conversion") from err
        elif rotation.shape == (4, 4):
            # Transformation matrix - validate and extract translation and rotation
            try:
                from scipy.spatial.transform import Rotation

                # Validate that the bottom row is [0, 0, 0, 1]
                if not np.allclose(rotation[3, :], [0, 0, 0, 1], atol=1e-6):
                    raise ValueError("Bottom row of 4x4 matrix must be [0, 0, 0, 1]")

                final_translation = rotation[:3, 3].copy()  # Extract translation
                rot_matrix = rotation[:3, :3]

                # Validate the rotation part
                if not np.allclose(np.dot(rot_matrix, rot_matrix.T), np.eye(3), atol=1e-6):
                    raise ValueError("Rotation part of 4x4 matrix is not orthogonal")
                if not np.isclose(np.linalg.det(rot_matrix), 1.0, atol=1e-6):
                    raise ValueError("Rotation part of 4x4 matrix has determinant != 1")

                quaternion = Rotation.from_matrix(rot_matrix).as_quat()  # Returns [x, y, z, w]
            except ImportError as err:
                raise ImportError("scipy is required for transformation matrix conversion") from err
        else:
            raise ValueError(f"Unsupported rotation format with shape {rotation.shape}")

        # Ensure final outputs are contiguous float arrays (backend safety)
        final_translation = np.ascontiguousarray(final_translation, dtype=np.float64)
        quaternion = np.ascontiguousarray(quaternion, dtype=np.float64)

        # Backend expects quaternion [x, y, z, w] and translation vector
        _STATE.backend.log_pose(full_path, final_translation, quaternion, **kwargs)  # type: ignore[union-attr]

    def set_time(self, timeline: str, sequence: int) -> None:
        """Set the current time sequence for the visualization timeline.

        Args:
            timeline: Name of the timeline (e.g., "step", "sample").
            sequence: Integer sequence number for this timeline.
        """
        if not _STATE.initialized:
            init()
        if not enabled():
            return
        if _STATE.backend and hasattr(_STATE.backend, "set_time"):
            _STATE.backend.set_time(timeline, sequence=sequence)

    def flush(self) -> None:
        # Flushing should still work even if sparse logging is temporarily disabled.
        if not _STATE.initialized or _STATE.backend is None:
            return
        _STATE.backend.flush()  # type: ignore[union-attr]

    def shutdown(self) -> None:
        """
        Perform cleanup during shutdown. Ensure the backend is active before shutting down.
        """
        # Important: shutdown must still finish the backend even if logging was temporarily disabled.
        if not _STATE.initialized:
            return
        try:
            if _STATE.backend is not None:
                # Only print if a backend was initialized
                if _STATE.backend_name != "disabled":
                    logger.info(f"{_VISUALIZER_LOG_PREFIX} Shutting down backend: {_STATE.backend_name}")
                _STATE.backend.shutdown()  # type: ignore[union-attr]
        finally:
            _STATE.enabled = False
            _STATE.backend = None
            _STATE.initialized = False


# Create a new DrakeVisualizer class for robot_gym and drake-specific methods
class DrakeVisualizer(Visualizer):
    """Extended visualizer with robot_gym and drake-specific logging methods."""

    @ensure_initialized_and_enabled
    def log_rigid_transform(self, path: str, transform: RigidTransform, **kwargs) -> None:
        """Log a rigid transform to the active backend.

        Args:
            path: Path in the visualization hierarchy.
            transform: Rigid transform object.
        """
        # Convert Drake RigidTransform to translation + quaternion for generic log_pose
        translation = transform.translation()
        rotation = transform.rotation().ToQuaternion()
        quaternion = np.array([rotation.x(), rotation.y(), rotation.z(), rotation.w()])  # [x, y, z, w]

        # Use the generic log_pose method
        self.log_pose(path, translation, quaternion, **kwargs)

    @ensure_initialized_and_enabled
    def log_robot_gym_poses_and_grippers(self, path: str, poses_and_grippers: PosesAndGrippers, **kwargs) -> None:
        """Log poses and grippers (robot-gym-specific) to the active backend.

        Args:
            path: Base path in the visualization hierarchy.
            poses_and_grippers: Object containing poses and gripper data.
        """

        for model_name, transform in poses_and_grippers.poses.items():
            self.log_rigid_transform(
                f"{path}/{model_name}",
                transform,
                axis_length=0.25,
                **kwargs,
            )

        if hasattr(poses_and_grippers, "grippers") and poses_and_grippers.grippers:
            for gripper_name, value in poses_and_grippers.grippers.items():
                self.log_scalar(f"{path}/{gripper_name}", value, **kwargs)

    @ensure_initialized_and_enabled
    def log_robot_gym_action_predictions(self, path: str, predictions: list[PosesAndGrippers], **kwargs) -> None:
        """Log action predictions (robot-gym-specific) to the active backend.

        Args:
            path: Base path in the visualization hierarchy.
            predictions: A list of objects containing action prediction data.
        """
        traj = {}
        for step in predictions:
            poses = getattr(step, "poses", {})
            for model_name, transform in poses.items():
                if model_name not in traj:
                    traj[model_name] = []
                traj[model_name].append(transform.translation())

        for model_name, points in traj.items():
            points_array = np.array(points)
            if len(points) >= 2:
                self.log_trajectory(f"{path}/{model_name}", points_array, **kwargs)
            elif points:
                self.log_points3d(f"{path}/{model_name}/waypoints", points_array, **kwargs)

    @ensure_initialized_and_enabled
    def log_robot_gym_multiarm_observation(self, path: str, observation: MultiarmObservation, **kwargs) -> None:
        """Log a MultiarmObservation (robot-gym-specific) to the active backend.

        Args:
            path: Base path in the visualization hierarchy.
            observation: The MultiarmObservation object to log.
        """
        self.log_robot_gym_poses_and_grippers(f"{path}/robot", observation.robot.actual, **kwargs)
        for camera_id, image_set in observation.visuo.items():
            if image_set.rgb:
                self.log_images(f"{path}/{camera_id}/rgb", image_set.rgb.array, **kwargs)
            if image_set.depth:
                self.log_images(f"{path}/{camera_id}/depth", image_set.depth.array, **kwargs)
            if image_set.label:
                self.log_images(f"{path}/{camera_id}/label", image_set.label.array, **kwargs)
        if observation.language_instruction:
            self.log_text(
                f"{path}/language_instruction",
                observation.language_instruction,
                **kwargs,
            )


# Default instances for facade-like usage
_default_visualizer = Visualizer()

# Lazy initialization for DrakeVisualizer
_drake_visualizer_instance = None


def _get_drake_visualizer() -> DrakeVisualizer:
    """
    Lazily initialize and return the DrakeVisualizer instance.
    """
    global _drake_visualizer_instance
    if _drake_visualizer_instance is None:
        _drake_visualizer_instance = DrakeVisualizer()
    return _drake_visualizer_instance


# Expose module-level functions for the default visualizer
log_images = _default_visualizer.log_images
# Removed log_image reference
log_scalar = _default_visualizer.log_scalar
log_points3d = _default_visualizer.log_points3d
log_trajectory = _default_visualizer.log_trajectory
log_line_strips3d = _default_visualizer.log_line_strips3d
log_text = _default_visualizer.log_text
log_pose = _default_visualizer.log_pose
set_time = _default_visualizer.set_time
flush = _default_visualizer.flush
shutdown = _default_visualizer.shutdown


# Expose Drake-specific functions with lazy initialization
def log_rigid_transform(path: str, transform: RigidTransform, **kwargs) -> None:
    _get_drake_visualizer().log_rigid_transform(path, transform, **kwargs)


def log_robot_gym_poses_and_grippers(path: str, poses_and_grippers: PosesAndGrippers, **kwargs) -> None:
    _get_drake_visualizer().log_robot_gym_poses_and_grippers(path, poses_and_grippers, **kwargs)


def log_robot_gym_action_predictions(path: str, predictions: list[PosesAndGrippers], **kwargs) -> None:
    _get_drake_visualizer().log_robot_gym_action_predictions(path, predictions, **kwargs)


def log_robot_gym_multiarm_observation(path: str, observation: MultiarmObservation, **kwargs) -> None:
    _get_drake_visualizer().log_robot_gym_multiarm_observation(path, observation, **kwargs)
