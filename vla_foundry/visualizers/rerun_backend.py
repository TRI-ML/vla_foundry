"""
Rerun Backend Implementation

This file contains the implementation of the RerunBackend class, which provides
logging functionality to rerun.io.
"""

import subprocess
from typing import Any

import numpy as np
import rerun as rr
from rerun import Transform3D
from rerun.datatypes import Quaternion


class RerunBackend:
    name = "rerun"

    def __init__(self) -> None:
        """
        Initialize the RerunBackend instance.
        """
        self._initialized = False
        self._disable_rerun_analytics()  # Ensure analytics are disabled during initialization

    def init(self, run_name: str, add_rank_to_run: bool = False, open_browser: bool = False, **kwargs) -> None:
        """
        Initialize the Rerun visualizer.

        Args:
            run_name: Name of the run.
            add_rank_to_run: Whether to add rank information to the run name.
            open_browser: Whether to automatically open the browser with the web viewer.
            **kwargs: Additional arguments (ignored).
        """
        if not self._initialized:
            # Detect if running in a notebook environment (Colab, Jupyter, JupyterLab)
            # These environments need spawn=False to keep recordings in-process
            is_notebook = False

            # Check for Google Colab specifically
            try:
                import importlib.util

                if importlib.util.find_spec("google.colab") is not None:
                    is_notebook = True
            except (ImportError, ValueError):
                pass

            # Check for general Jupyter/IPython kernel environment if not Colab
            if not is_notebook:
                try:
                    from IPython import get_ipython

                    ipython = get_ipython()
                    if ipython is not None and hasattr(ipython, "config"):
                        # Running in IPython kernel (Jupyter, JupyterLab, etc.)
                        is_notebook = "IPKernelApp" in ipython.config
                except (ImportError, AttributeError):
                    pass

            if is_notebook:
                rr.init(run_name, spawn=False)
                self._initialized = True
            else:
                rr.init(run_name)
                server_uri = rr.serve_grpc()
                rr.serve_web_viewer(open_browser=open_browser, web_port=9090, connect_to=server_uri)
                encoded_uri = server_uri.replace("+", "%2B")
                print(f"[rerun_backend] Open in browser: http://localhost:9090/?url={encoded_uri}")
                self._initialized = True

    def flush(self) -> None:
        # rerun flush is implicit; no-op here
        return

    def shutdown(self) -> None:
        # rerun doesn't strictly need it; keep for parity
        return

    def _disable_rerun_analytics(self) -> None:
        try:
            subprocess.run(
                ["rerun", "analytics", "disable"],
                check=True,
                capture_output=True,
                text=True,
            )
            print("[rerun_backend] Rerun analytics disabled.")
        except FileNotFoundError:
            print("[rerun_backend] rerun CLI not found; analytics may be enabled. Ensure the rerun CLI is installed.")
        except subprocess.CalledProcessError as e:
            print(f"[rerun_backend] Failed to disable analytics: {e.stderr or e}")

    def log_images(self, path: str, images: Any, **kwargs) -> None:
        """
        Log images to the rerun backend. Supports both single images and dictionaries of images.

        Args:
            path: Base path in the visualization hierarchy.
            images: Either a single NumPy array representing an image or a dictionary of images.
        """
        if isinstance(images, np.ndarray):
            # Single image
            rr.log(path, rr.Image(images))  # Updated to use rr.Image for logging
        elif isinstance(images, dict):
            # Dictionary of images
            for sub_path, image in images.items():
                rr.log(f"{path}/{sub_path}", rr.Image(image))  # Updated to use rr.Image for logging
        else:
            raise TypeError("Unsupported type for 'images'. Must be a NumPy array or a dictionary of NumPy arrays.")

    def log_scalar(self, path: str, value: float, **kwargs) -> None:
        """
        Log a scalar value to the Rerun backend.

        Args:
            path: The hierarchical path for the scalar.
            value: The scalar value.
        """
        rr.log(path, rr.Scalars(value))

    def log_points3d(
        self, path: str, points: np.ndarray, radii: np.ndarray = None, colors: np.ndarray = None, **kwargs
    ) -> None:
        """
        Log 3D points to the Rerun backend.

        Args:
            path: The hierarchical path for the 3D points.
            points: The 3D points as a NumPy array of shape (N, 3).
            radii: Optional array of point radii.
            colors: Optional array of point colors.
        """
        rr.log(path, rr.Points3D(points, radii=radii, colors=colors))

    def log_trajectory(self, path: str, trajectory: np.ndarray, **kwargs) -> None:
        """
        Log a trajectory to the Rerun backend.

        Args:
            path: The hierarchical path for the trajectory.
            trajectory: The trajectory as a NumPy array of shape (N, 3).
        """
        rr.log(path, rr.LineStrips3D([trajectory]))
        rr.log(f"{path}/waypoints", rr.Points3D(trajectory))

    def log_pose(
        self, path: str, translation: np.ndarray, rotation: np.ndarray, axis_length: float = 1.0, **kwargs
    ) -> None:
        """
        Log a generic pose to the Rerun backend.

        Args:
            path: Path in the visualization hierarchy.
            translation: Translation vector of shape (3,).
            rotation: Quaternion [x, y, z, w] of shape (4,).
            axis_length: Length of the axes for visualization, by default 1.0.
        """
        rr.log(
            path,
            Transform3D(
                translation=translation,
                rotation=Quaternion(xyzw=rotation),
            ),
        )

    def log_line_strips3d(self, path: str, line_strips: np.ndarray | dict[str, np.ndarray], **kwargs) -> None:
        """
        Log 3D line strips to the Rerun backend.

        Parameters
        ----------
        path : str
            The hierarchical path for the line strips.
        line_strips : np.ndarray | dict[str, np.ndarray]
            The 3D line strips as a NumPy array of shape (N, 3),
            or multiple named line strips as a dictionary.
        """
        if isinstance(line_strips, np.ndarray):
            rr.log(path, rr.LineStrips3D([line_strips]))
            return
        if isinstance(line_strips, dict):
            for name, points in line_strips.items():
                rr.log(f"{path}/{name}", rr.LineStrips3D([np.asarray(points)]))
            return
        raise TypeError("line_strips must be np.ndarray or dict[str, np.ndarray]")

    def log_text(self, path: str, text: str, **kwargs) -> None:
        """
        Log a text value to the Rerun backend.

        Args:
            path: Path in the visualization hierarchy.
            text: Text value to log.
        """
        rr.log(path, rr.TextLog(text))

    def set_time(self, timeline: str, sequence: int) -> None:
        """
        Set the current time for the timeline.

        Args:
            timeline: Name of the timeline (e.g., "step", "sample").
            sequence: Integer sequence number for this timeline.
        """
        rr.set_time(timeline, sequence=sequence)
