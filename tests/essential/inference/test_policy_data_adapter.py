#!/usr/bin/env python3
"""
Tests for PolicyDataAdapter aligned with the current refactored implementation.
"""

import sys
import tempfile
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml


# We don't want to depend on pydrake for testing
class RotationMatrix:
    def __init__(self, matrix: "np.ndarray | RotationMatrix | None" = None):
        if isinstance(matrix, RotationMatrix):
            matrix = matrix.matrix()
        if matrix is None:
            matrix = np.eye(3, dtype=np.float64)
        self._matrix = np.asarray(matrix, dtype=np.float64)

    @classmethod
    def MakeZRotation(cls, angle: float) -> "RotationMatrix":
        cosine = np.cos(angle)
        sine = np.sin(angle)
        return cls(
            np.array(
                [
                    [cosine, -sine, 0.0],
                    [sine, cosine, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        )

    def matrix(self) -> np.ndarray:
        return self._matrix


class RigidTransform:
    def __init__(
        self,
        rotation: RotationMatrix | np.ndarray | None = None,
        translation: np.ndarray | None = None,
        *,
        R: RotationMatrix | np.ndarray | None = None,
        p: np.ndarray | None = None,
    ):
        if rotation is None:
            rotation = R
        if translation is None:
            translation = p

        if isinstance(rotation, np.ndarray):
            rotation = RotationMatrix(rotation)
        if rotation is None:
            rotation = RotationMatrix()
        if translation is None:
            translation = np.zeros(3, dtype=np.float64)

        self._rotation = rotation
        self._translation = np.asarray(translation, dtype=np.float64)

    def rotation(self) -> RotationMatrix:
        return self._rotation

    def translation(self) -> np.ndarray:
        return self._translation


pydrake_module = sys.modules.get("pydrake")
if pydrake_module is None:
    pydrake_module = ModuleType("pydrake")
    sys.modules["pydrake"] = pydrake_module
math_module = ModuleType("pydrake.math")
math_module.RigidTransform = RigidTransform
math_module.RotationMatrix = RotationMatrix
pydrake_module.math = math_module
sys.modules["pydrake.math"] = math_module


# We don't want to depend on robot_gym for testing
@dataclass
class CameraRgbImage:
    array: np.ndarray
    K: np.ndarray
    X_TC: RigidTransform


@dataclass
class CameraImageSet:
    rgb: CameraRgbImage | None = None
    depth: np.ndarray | None = None
    label: np.ndarray | None = None


@dataclass
class PosesAndGrippers:
    poses: dict[str, RigidTransform]
    grippers: dict[str, float]
    joint_position: dict[str, np.ndarray] = dataclass_field(default_factory=dict)


@dataclass
class PosesAndGrippersActualAndDesired:
    actual: PosesAndGrippers
    desired: PosesAndGrippers


@dataclass
class MultiarmObservation:
    robot: PosesAndGrippersActualAndDesired
    visuo: dict[str, CameraImageSet]
    language_instruction: str
    timestamp_packaged: float


robot_gym_module = sys.modules.get("robot_gym")
if robot_gym_module is None:
    robot_gym_module = ModuleType("robot_gym")
    sys.modules["robot_gym"] = robot_gym_module
multiarm_module = ModuleType("robot_gym.multiarm_spaces")
multiarm_module.CameraImageSet = CameraImageSet
multiarm_module.CameraRgbImage = CameraRgbImage
multiarm_module.MultiarmObservation = MultiarmObservation
multiarm_module.PosesAndGrippers = PosesAndGrippers
multiarm_module.PosesAndGrippersActualAndDesired = PosesAndGrippersActualAndDesired
robot_gym_module.multiarm_spaces = multiarm_module
sys.modules["robot_gym.multiarm_spaces"] = multiarm_module


from vla_foundry.data.robotics.utils import (  # noqa: E402
    apply_relative_pose,
    calculate_relative_pose,
    pose_to_9d,
    rot_6d_to_matrix,
    to_pose_matrix,
)
from vla_foundry.inference.robotics.data_adapter import (  # noqa: E402
    PolicyDataAdapter,
)
from vla_foundry.inference.robotics.lbm_mapping import ObservationMapping  # noqa: E402
from vla_foundry.inference.robotics.utils import (  # noqa: E402
    any_to_actual_map,
    relative_to_absolute_map,
)


@pytest.fixture
def mock_robotics_processor():
    """Create a mock robotics processor with required interfaces."""
    processor = Mock()

    processor.get_field_dimension = Mock(
        side_effect=lambda field: 3 if "xyz" in field else 6 if "rot_6d" in field else 1
    )
    processor.get_timestep_dimension = Mock(return_value=26)

    normalizer = Mock()

    normalizer.normalize_tensor = Mock(side_effect=lambda tensor, field, anchor_timestep=None: tensor)
    normalizer.denormalize_tensor = Mock(side_effect=lambda tensor, field, anchor_timestep=None: tensor)
    normalizer.get_field_dimension = Mock(side_effect=lambda field: processor.get_field_dimension(field))
    action_fields = [
        "robot__action__poses__left::panda__xyz",
        "robot__action__poses__right::panda__xyz",
        "robot__action__poses__left::panda__rot_6d",
        "robot__action__poses__right::panda__rot_6d",
        "robot__action__grippers__left::panda_hand",
        "robot__action__grippers__right::panda_hand",
    ]

    def _make_stats_entry(dim: int, timesteps: int = 32) -> dict:
        return {
            "std": [0.1] * dim,
            "std_per_timestep": [[0.1] * dim for _ in range(timesteps)],
            "min": [0.0] * dim,
            "max": [1.0] * dim,
        }

    normalizer.stats = {}
    for action_field in action_fields:
        dim = processor.get_field_dimension(action_field)
        if "gripper" in action_field:
            # Use min/max matching actual test gripper values (0.2 closed, 0.4 open)
            stats = _make_stats_entry(dim)
            stats["min"] = [0.2] * dim
            stats["max"] = [0.4] * dim
            normalizer.stats[action_field] = stats
        else:
            normalizer.stats[action_field] = _make_stats_entry(dim)
        actual_field = action_field.replace("robot__action__", "robot__actual__")
        normalizer.stats[actual_field] = normalizer.stats[action_field]

    processor.normalizer = normalizer

    def _process_inputs(batch, image_names):
        lowdim_raw = batch["lowdim"][0]
        normalized_lowdim = {
            field: processor.normalizer.normalize_tensor(tensor, field) for field, tensor in lowdim_raw.items()
        }
        num_images = len(image_names)
        return {
            "input_ids": torch.randint(0, 1000, (1, 512)),
            "attention_mask": torch.ones(1, 512, dtype=torch.bool),
            "pixel_values": torch.randn(1, num_images, 3, 224, 224),
            "language_instruction": ["processed_instruction"],
            "lowdim": normalized_lowdim,
        }

    processor.process_inputs = Mock(side_effect=_process_inputs)

    def _add_action_and_proprioception_fields(batch, action_fields=None, proprioception_fields=None):
        action_fields = action_fields or []
        proprioception_fields = proprioception_fields or []
        actions = [batch["lowdim"][field] for field in action_fields if field in batch["lowdim"]]
        if actions:
            actions_tensor = torch.cat(actions, dim=-1)
            if actions_tensor.ndim == 2:
                actions_tensor = actions_tensor.unsqueeze(0)
            batch["actions"] = actions_tensor
        proprio = [batch["lowdim"][field] for field in proprioception_fields if field in batch["lowdim"]]
        if proprio:
            proprio_tensor = torch.cat(proprio, dim=-1)
            if proprio_tensor.ndim == 2:
                proprio_tensor = proprio_tensor.unsqueeze(0)
            batch["proprioception"] = proprio_tensor
        return batch

    processor.add_action_and_proprioception_fields = Mock(side_effect=_add_action_and_proprioception_fields)

    return processor


@pytest.fixture
def mock_data_config():
    """Create a minimal data configuration compatible with PolicyDataAdapter."""
    action_fields = [
        "robot__action__poses__left::panda__xyz",
        "robot__action__poses__right::panda__xyz",
        "robot__action__poses__left::panda__rot_6d",
        "robot__action__poses__right::panda__rot_6d",
        "robot__action__grippers__left::panda_hand",
        "robot__action__grippers__right::panda_hand",
    ]

    image_names = [
        "wrist_left_minus_t-1",
        "wrist_left_minus_t0",
        "wrist_left_plus_t-1",
        "wrist_left_plus_t0",
        "wrist_right_minus_t-1",
        "wrist_right_minus_t0",
        "wrist_right_plus_t-1",
        "wrist_right_plus_t0",
        "scene_left_0_t-1",
        "scene_left_0_t0",
        "scene_right_0_t-1",
        "scene_right_0_t0",
    ]

    augmentation = SimpleNamespace(image=SimpleNamespace(crop=SimpleNamespace(shape=(224, 224))))

    return SimpleNamespace(
        action_fields=action_fields,
        image_names=image_names,
        augmentation=augmentation,
        image_size=224,
        proprioception_fields=[],
        pose_groups=[],
    )


@pytest.fixture
def field_mapping_file():
    """Create a temporary YAML field mapping file."""
    field_mapping = {
        "field_paths": {
            "robot__action__poses__left::panda__xyz": ["left::panda", "translation"],
            "robot__action__poses__right::panda__xyz": ["right::panda", "translation"],
            "robot__action__poses__left::panda__rot_6d": ["left::panda", "rotation_6d"],
            "robot__action__poses__right::panda__rot_6d": ["right::panda", "rotation_6d"],
            "robot__action__grippers__left::panda_hand": ["left::panda_hand"],
            "robot__action__grippers__right::panda_hand": ["right::panda_hand"],
            "robot__actual__poses__left::panda__xyz": ["left::panda", "translation"],
            "robot__actual__poses__right::panda__xyz": ["right::panda", "translation"],
            "robot__actual__poses__left::panda__rot_6d": ["left::panda", "rotation_6d"],
            "robot__actual__poses__right::panda__rot_6d": ["right::panda", "rotation_6d"],
            "robot__actual__joint_position__left::panda": ["left::panda"],
            "robot__actual__joint_position__right::panda": ["right::panda"],
            "robot__actual__grippers__left::panda_hand": ["left::panda_hand"],
            "robot__actual__grippers__right::panda_hand": ["right::panda_hand"],
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(field_mapping, handle)
        return handle.name


class TrackingNormalizer:
    def __init__(self, field_dims: dict[str, int] | None = None):
        self.normalize_calls = []
        self.denormalize_calls = []
        # Mock stats for testing
        self.stats = {
            "robot__action__poses__left::panda__xyz": {
                "std": [0.1, 0.1, 0.1],
                "std_per_timestep": [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            },
            "robot__action__poses__left::panda__rot_6d": {
                "std": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                "std_per_timestep": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
                "min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            "robot__action__grippers__left::panda_hand": {
                "std": [0.1],
                "std_per_timestep": [[0.1], [0.1]],
                "min": [0.0],
                "max": [1.0],
            },
        }
        self._field_dims = field_dims or {}

    def normalize_tensor(self, tensor: torch.Tensor, field: str, anchor_timestep: int | None = None) -> torch.Tensor:
        self.normalize_calls.append((field, tensor.clone()))
        return (tensor + 1.0) / 2.0

    def denormalize_tensor(self, tensor: torch.Tensor, field: str, anchor_timestep: int | None = None) -> torch.Tensor:
        self.denormalize_calls.append((field, tensor.clone()))
        return tensor * 2.0 - 1.0

    def get_field_dimension(self, field: str) -> int:
        if field in self._field_dims:
            return self._field_dims[field]
        if field in self.stats and "std" in self.stats[field]:
            return len(self.stats[field]["std"])
        raise KeyError(f"Unknown field dimension for '{field}'")


def _normalize_ref(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor + 1.0) / 2.0


def _denormalize_ref(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2.0 - 1.0


class FakeRoboticsProcessor:
    def __init__(self, field_dims: dict[str, int], timestep_dim: int):
        self._field_dims = field_dims
        self._timestep_dim = timestep_dim
        self.normalizer = TrackingNormalizer(field_dims)
        self.process_inputs_calls = []

    def get_field_dimension(self, field: str) -> int:
        return self._field_dims[field]

    def get_timestep_dimension(self) -> int:
        return self._timestep_dim

    def process_inputs(self, batch: dict, image_names: list[str]) -> dict:
        self.process_inputs_calls.append((batch, image_names))
        num_images = len(image_names)
        lowdim_raw = batch["lowdim"][0]
        normalized_lowdim = {
            field: self.normalizer.normalize_tensor(tensor, field) for field, tensor in lowdim_raw.items()
        }
        return {
            "input_ids": torch.arange(12, dtype=torch.long).view(1, 12),
            "attention_mask": torch.ones(1, 12, dtype=torch.bool),
            "pixel_values": torch.ones(1, num_images, 3, 112, 112),
            "language_instruction": batch["language_instruction"],
            "lowdim": normalized_lowdim,
        }

    def add_action_and_proprioception_fields(self, batch: dict, action_fields=None, proprioception_fields=None) -> dict:
        action_fields = action_fields or []
        proprioception_fields = proprioception_fields or []
        actions = [batch["lowdim"][field] for field in action_fields if field in batch["lowdim"]]
        if actions:
            actions_tensor = torch.cat(actions, dim=-1)
            if actions_tensor.ndim == 2:
                actions_tensor = actions_tensor.unsqueeze(0)
            batch["actions"] = actions_tensor
        proprio = [batch["lowdim"][field] for field in proprioception_fields if field in batch["lowdim"]]
        if proprio:
            proprio_tensor = torch.cat(proprio, dim=-1)
            if proprio_tensor.ndim == 2:
                proprio_tensor = proprio_tensor.unsqueeze(0)
            batch["proprioception"] = proprio_tensor
        return batch


@pytest.fixture
def create_multiarm_observation():
    """Factory to create MultiarmObservation instances with deterministic data."""

    def _create(step: int, camera_names: list[str]):
        left_pose = RigidTransform(
            RotationMatrix(np.eye(3)),
            np.array([0.1 + step * 0.01, 0.2 + step * 0.01, 0.3 + step * 0.01]),
        )
        right_pose = RigidTransform(
            RotationMatrix(np.eye(3)),
            np.array([-0.4 - step * 0.01, 0.5 + step * 0.01, -0.6 - step * 0.01]),
        )

        left_gripper = 0.02 + step * 0.001
        right_gripper = 0.04 + step * 0.001

        left_joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) + step * 0.01
        right_joints = np.array([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7]) - step * 0.01

        visuo = {}
        for name in camera_names:
            visuo[name] = CameraImageSet(
                rgb=CameraRgbImage(
                    array=np.full((224, 224, 3), step * 10, dtype=np.uint8),
                    K=np.eye(3),
                    X_TC=RigidTransform(),
                ),
                depth=None,
                label=None,
            )

        actual = PosesAndGrippers(
            poses={"left::panda": left_pose, "right::panda": right_pose},
            grippers={"left::panda_hand": left_gripper, "right::panda_hand": right_gripper},
            joint_position={"left::panda": left_joints, "right::panda": right_joints},
        )

        desired = PosesAndGrippers(
            poses={"left::panda": left_pose, "right::panda": right_pose},
            grippers={"left::panda_hand": left_gripper, "right::panda_hand": right_gripper},
            joint_position={"left::panda": left_joints, "right::panda": right_joints},
        )

        robot = PosesAndGrippersActualAndDesired(actual=actual, desired=desired)

        return MultiarmObservation(
            robot=robot,
            visuo=visuo,
            language_instruction=f"Instruction step {step}",
            timestamp_packaged=step * 0.1,
        )

    return _create


def test_observation_mapping_joint_position(field_mapping_file, create_multiarm_observation):
    observation = create_multiarm_observation(step=0, camera_names=[])
    mapping = ObservationMapping(field_mapping_file, image_names=[])

    result = mapping.get_field(observation, "robot__actual__joint_position__left::panda")
    np.testing.assert_allclose(result, observation.robot.actual.joint_position["left::panda"])


def _camera_basenames(image_names: list[str]) -> list[str]:
    return sorted({name.rsplit("_t", 1)[0] for name in image_names})


def test_policy_data_adapter_end_to_end_flow(field_mapping_file):
    """Verify buffers, normalization, and conversions across a full adapter cycle."""

    num_past = 2
    num_future = 1
    image_indices = (-1, 0)
    image_names = ["cam0_t-1", "cam0_t0"]
    action_fields = [
        "robot__action__poses__left::panda__xyz_relative",
        "robot__action__poses__left::panda__rot_6d_relative",
        "robot__action__grippers__left::panda_hand",
    ]
    pose_groups = [
        {
            "name": "left_panda_action",
            "position_key": "robot__action__poses__left::panda__xyz",
            "rotation_key": "robot__action__poses__left::panda__rot_6d",
        }
    ]
    field_dims = {
        action_fields[0]: 3,
        action_fields[1]: 6,
        action_fields[2]: 1,
    }

    processor = FakeRoboticsProcessor(field_dims=field_dims, timestep_dim=num_past + 1 + num_future)

    augmentation = SimpleNamespace(image=SimpleNamespace(crop=SimpleNamespace(shape=(112, 112))))
    data_config = SimpleNamespace(
        action_fields=action_fields,
        image_names=image_names,
        augmentation=augmentation,
        proprioception_fields=[],
        pose_groups=pose_groups,
    )

    adapter = PolicyDataAdapter(
        robotics_processor=processor,
        data_config=data_config,
        field_mapping_path=field_mapping_file,
        image_names=image_names,
        preprocessor_image_size=(128, 128),
        num_past_timesteps=num_past,
        num_future_timesteps=num_future,
        image_indices=image_indices,
    )

    def make_observation(
        step: int, translation: np.ndarray, rotation: RotationMatrix, gripper: float, image_value: int
    ):
        image = np.full((150, 150, 3), image_value, dtype=np.uint8)
        visuo = {
            "cam0": CameraImageSet(
                rgb=CameraRgbImage(array=image, K=np.eye(3), X_TC=RigidTransform()),
                depth=None,
                label=None,
            )
        }

        actual = PosesAndGrippers(
            poses={"left::panda": RigidTransform(rotation, translation)},
            grippers={"left::panda_hand": gripper},
            joint_position={"left::panda": np.zeros(7)},
        )
        desired = PosesAndGrippers(
            poses={"left::panda": RigidTransform(rotation, translation)},
            grippers={"left::panda_hand": gripper},
            joint_position={"left::panda": np.zeros(7)},
        )
        robot = PosesAndGrippersActualAndDesired(actual=actual, desired=desired)

        return MultiarmObservation(
            robot=robot,
            visuo=visuo,
            language_instruction=f"Instruction step {step}",
            timestamp_packaged=step * 0.1,
        )

    translation0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    rotation0 = RotationMatrix.MakeZRotation(0.0)
    gripper0 = 0.2

    translation1 = np.array([0.5, -0.2, 0.3], dtype=np.float64)
    rotation1 = RotationMatrix.MakeZRotation(0.45)
    gripper1 = 0.4

    obs0 = make_observation(step=0, translation=translation0, rotation=rotation0, gripper=gripper0, image_value=10)
    obs1 = make_observation(step=1, translation=translation1, rotation=rotation1, gripper=gripper1, image_value=20)

    adapter.reset(obs0)

    expected_total = num_past + 1 + num_future

    assert len(adapter.action_buffer) == expected_total
    np.testing.assert_allclose(adapter.action_buffer[0]["robot__action__poses__left::panda__xyz"], translation0)
    np.testing.assert_allclose(adapter.reference["robot__actual__poses__left::panda__xyz"], translation0)

    adapter.step_observations(obs1)
    executed_action = adapter.step_action()
    assert isinstance(executed_action, PosesAndGrippers)
    np.testing.assert_allclose(executed_action.poses["left::panda"].translation(), translation0, atol=1e-6)
    # gripper0=0.2 normalizes to 0.2 in [0,1] (stats min=0.0, max=1.0),
    # which is below close_threshold=0.4, so debouncer snaps to min (closed)
    gripper_stats = processor.normalizer.stats["robot__action__grippers__left::panda_hand"]
    assert executed_action.grippers["left::panda_hand"] == pytest.approx(gripper_stats["min"][0], abs=1e-6)

    for idx in range(adapter.num_past_timesteps):
        np.testing.assert_allclose(
            adapter.action_buffer[idx]["robot__action__poses__left::panda__xyz"],
            translation0,
        )
    assert adapter.past_mask.shape == (1, num_past + 1 + num_future)
    assert adapter.past_mask[0, num_past - 1]
    assert not adapter.past_mask[0, num_past]

    processor.process_inputs_calls.clear()
    processor.normalizer.normalize_calls.clear()
    processor.normalizer.denormalize_calls.clear()

    with np.errstate(invalid="ignore"):
        model_input = adapter.get_model_input(obs1)

    assert processor.process_inputs_calls
    batch, image_names_called = processor.process_inputs_calls[-1]
    assert image_names_called == image_names
    images_dict = batch["images"][0]
    assert set(images_dict.keys()) == set(image_names)
    np.testing.assert_array_equal(images_dict["cam0_t-1"], np.full((150, 150, 3), 10, dtype=np.uint8))
    np.testing.assert_array_equal(images_dict["cam0_t0"], np.full((112, 112, 3), 20, dtype=np.uint8))
    assert batch["language_instruction"] == ["Instruction step 1"]

    # Set up field mappings for reuse
    reference_index = min(adapter.num_past_timesteps - 1, len(adapter.action_buffer) - 1)
    absolute_xyz_field = relative_to_absolute_map(action_fields[0])
    absolute_rot_field = relative_to_absolute_map(action_fields[1])
    absolute_actual_xyz_field = any_to_actual_map(absolute_xyz_field)
    absolute_actual_rot_field = any_to_actual_map(absolute_rot_field)

    # Stack action buffer data for processing
    stacked_xyz = np.stack(
        [
            adapter.action_buffer[idx]["robot__action__poses__left::panda__xyz"]
            for idx in range(len(adapter.action_buffer))
        ],
        axis=0,
    )
    stacked_rot6d = np.stack(
        [
            adapter.action_buffer[idx]["robot__action__poses__left::panda__rot_6d"]
            for idx in range(len(adapter.action_buffer))
        ],
        axis=0,
    )

    # Convert to relative using pose matrices
    reference_xyz = adapter.reference[absolute_actual_xyz_field]
    reference_rot = adapter.reference[absolute_actual_rot_field]
    reference_pose_matrix = to_pose_matrix(reference_xyz, reference_rot)
    action_pose_matrices = to_pose_matrix(stacked_xyz, stacked_rot6d)
    relative_pose_matrices = calculate_relative_pose(action_pose_matrices, reference_pose_matrix)
    expected_xyz_relative, expected_rot_relative = pose_to_9d(relative_pose_matrices)

    expected_xyz_tensor = torch.tensor(expected_xyz_relative, dtype=torch.float32)
    expected_rot_tensor = torch.tensor(expected_rot_relative, dtype=torch.float32)

    stacked_gripper = np.stack(
        [
            adapter.action_buffer[idx]["robot__action__grippers__left::panda_hand"]
            for idx in range(len(adapter.action_buffer))
        ],
        axis=0,
    )
    expected_gripper_tensor = torch.tensor(stacked_gripper[:, None], dtype=torch.float32)

    lowdim = batch["lowdim"][0]
    torch.testing.assert_close(lowdim[action_fields[0]], expected_xyz_tensor)
    torch.testing.assert_close(lowdim[action_fields[1]], expected_rot_tensor)
    torch.testing.assert_close(lowdim[action_fields[2]], expected_gripper_tensor)

    expected_fields = action_fields
    assert [field for field, _ in processor.normalizer.normalize_calls] == expected_fields
    expected_tensors = [expected_xyz_tensor, expected_rot_tensor, expected_gripper_tensor]
    for idx, expected_tensor in enumerate(expected_tensors):
        torch.testing.assert_close(processor.normalizer.normalize_calls[idx][1], expected_tensor)

    expected_action_for_model = torch.cat(
        [
            _normalize_ref(expected_xyz_tensor),
            _normalize_ref(expected_rot_tensor),
            _normalize_ref(expected_gripper_tensor),
        ],
        dim=-1,
    ).unsqueeze(0)
    torch.testing.assert_close(model_input["actions"], expected_action_for_model)
    assert model_input["actions"].shape == (
        1,
        num_past + 1 + num_future,
        sum(field_dims.values()),
    )

    actions_tensor = model_input["actions"].clone()
    model_output = actions_tensor.clone()
    future_relative_xyz = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float32)
    xyz_slice = slice(0, 3)
    rot_slice = slice(3, 9)
    gripper_index = 9

    model_output[0, num_past, xyz_slice] = _normalize_ref(future_relative_xyz)
    identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    model_output[0, num_past, rot_slice] = _normalize_ref(identity_rot6d)
    model_output[0, num_past, gripper_index] = _normalize_ref(torch.tensor(0.55, dtype=torch.float32))

    with np.errstate(invalid="ignore"):
        adapter.update_action(obs1, model_output)

    actions = [adapter.action_mapping.create_pose_and_gripper(entry) for entry in adapter.action_buffer]

    assert len(actions) == model_output.shape[1]
    assert len(adapter.action_buffer) == expected_total
    future_start = num_past
    np.testing.assert_allclose(
        adapter.action_buffer[future_start]["robot__action__poses__left::panda__xyz"],
        actions[future_start].poses["left::panda"].translation(),
    )
    assert [field for field, _ in processor.normalizer.denormalize_calls] == action_fields
    torch.testing.assert_close(processor.normalizer.denormalize_calls[0][1], model_output[:, :, xyz_slice])

    for idx, pose_action in enumerate(actions):
        for action_field in adapter.action_fields:
            absolute_field = relative_to_absolute_map(action_field)
            mapping_fields = adapter.action_mapping._field_paths[absolute_field]
            actual_value = adapter.action_buffer[idx][absolute_field]
            if "rot_6d" in action_field:
                actual_matrix = rot_6d_to_matrix(actual_value)
                expected_matrix = pose_action.poses[mapping_fields[0]].rotation().matrix()
                np.testing.assert_allclose(actual_matrix, expected_matrix, atol=1e-6)
            elif "xyz" in action_field:
                expected_translation = pose_action.poses[mapping_fields[0]].translation()
                np.testing.assert_allclose(actual_value, expected_translation)
            elif "gripper" in action_field:
                expected_gripper = pose_action.grippers[mapping_fields[0]]
                np.testing.assert_allclose(actual_value, expected_gripper)

    # Verify reference values (field mappings already computed above)
    action_reference = adapter.action_buffer[reference_index][absolute_xyz_field]
    action_reference_rot = adapter.action_buffer[reference_index][absolute_rot_field]
    actual_reference = adapter.reference[absolute_actual_xyz_field]
    actual_reference_rot = adapter.reference[absolute_actual_rot_field]

    np.testing.assert_allclose(action_reference, translation0, atol=1e-6)
    np.testing.assert_allclose(action_reference_rot, stacked_rot6d[reference_index], atol=1e-6)
    np.testing.assert_allclose(actual_reference, translation1, atol=1e-6)
    rotation1_6d = rotation1.matrix()[:2, :].flatten()
    np.testing.assert_allclose(actual_reference_rot, rotation1_6d, atol=1e-6)

    # Convert model output from relative to absolute poses
    relative_xyz = _denormalize_ref(model_output[0, num_past, xyz_slice])
    relative_rot = _denormalize_ref(model_output[0, num_past, rot_slice])
    reference_pose_matrix = to_pose_matrix(actual_reference, actual_reference_rot)
    relative_pose_matrix = to_pose_matrix(relative_xyz.cpu().numpy(), relative_rot.cpu().numpy())
    absolute_pose_matrix = apply_relative_pose(relative_pose_matrix, reference_pose_matrix)
    expected_translation, expected_rot6d = pose_to_9d(absolute_pose_matrix)
    expected_rot_matrix = rot_6d_to_matrix(expected_rot6d)

    np.testing.assert_allclose(actions[num_past].poses["left::panda"].translation(), expected_translation, atol=1e-6)
    np.testing.assert_allclose(
        actions[num_past].poses["left::panda"].rotation().matrix(), expected_rot_matrix, atol=1e-6
    )
    assert actions[num_past].grippers["left::panda_hand"] == pytest.approx(0.55, abs=1e-6)

    np.testing.assert_allclose(adapter.reference["robot__actual__poses__left::panda__xyz"], translation1)


def test_policy_data_adapter_open_loop_cycle(field_mapping_file):
    """Simulate repeated inference replans and verify action buffer rollouts."""

    num_past = 2
    num_future = 3
    open_loop_steps = 3
    image_indices = (-1, 0)
    image_names = ["cam0_t-1", "cam0_t0"]
    action_fields = [
        "robot__action__poses__left::panda__xyz_relative",
        "robot__action__poses__left::panda__rot_6d_relative",
        "robot__action__grippers__left::panda_hand",
    ]
    pose_groups = [
        {
            "name": "left_panda_action",
            "position_key": "robot__action__poses__left::panda__xyz",
            "rotation_key": "robot__action__poses__left::panda__rot_6d",
        }
    ]
    field_dims = {
        action_fields[0]: 3,
        action_fields[1]: 6,
        action_fields[2]: 1,
    }

    processor = FakeRoboticsProcessor(field_dims=field_dims, timestep_dim=num_past + 1 + num_future)

    augmentation = SimpleNamespace(image=SimpleNamespace(crop=SimpleNamespace(shape=(112, 112))))
    data_config = SimpleNamespace(
        action_fields=action_fields,
        image_names=image_names,
        augmentation=augmentation,
        proprioception_fields=[],
        pose_groups=pose_groups,
    )

    adapter = PolicyDataAdapter(
        robotics_processor=processor,
        data_config=data_config,
        field_mapping_path=field_mapping_file,
        image_names=image_names,
        preprocessor_image_size=(128, 128),
        num_past_timesteps=num_past,
        num_future_timesteps=num_future,
        image_indices=image_indices,
    )

    def make_observation(step: int) -> MultiarmObservation:
        base_translation = np.array([0.2 - 0.01 * step, -0.3 + 0.02 * step, 0.4 + 0.015 * step])
        base_rotation = RotationMatrix.MakeZRotation(0.05 * step)
        base_gripper = 0.3 + 0.01 * step

        visuo = {
            "cam0": CameraImageSet(
                rgb=CameraRgbImage(
                    array=np.full((128, 128, 3), 10 + step, dtype=np.uint8), K=np.eye(3), X_TC=RigidTransform()
                ),
                depth=None,
                label=None,
            )
        }

        actual = PosesAndGrippers(
            poses={"left::panda": RigidTransform(base_rotation, base_translation)},
            grippers={"left::panda_hand": base_gripper},
            joint_position={"left::panda": np.zeros(7)},
        )
        desired = PosesAndGrippers(
            poses={"left::panda": RigidTransform(base_rotation, base_translation)},
            grippers={"left::panda_hand": base_gripper},
            joint_position={"left::panda": np.zeros(7)},
        )

        robot = PosesAndGrippersActualAndDesired(actual=actual, desired=desired)

        return MultiarmObservation(
            robot=robot,
            visuo=visuo,
            language_instruction=f"Instruction step {step}",
            timestamp_packaged=step * 0.1,
        )

    total_steps = 12
    observations = [make_observation(step) for step in range(total_steps)]

    adapter.reset(observations[0])

    executed_translations: list[np.ndarray] = []
    total_replans = 0
    open_loop_counter = 0
    last_replan_translation = None

    for step_idx in range(1, total_steps):
        obs = observations[step_idx]
        need_replan = open_loop_counter % open_loop_steps == 0

        if need_replan:
            with np.errstate(invalid="ignore"):
                model_input = adapter.get_model_input(obs)
            model_output = model_input["actions"].clone()
            identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
            normalized_rot = _normalize_ref(identity_rot6d).unsqueeze(0).repeat(model_output.shape[1], 1)
            model_output[0, :, 3:9] = normalized_rot
            with np.errstate(invalid="ignore"):
                adapter.update_action(obs, model_output)
            predicted_actions = [
                adapter.action_mapping.create_pose_and_gripper(entry) for entry in adapter.action_buffer
            ]
            last_replan_translation = obs.robot.actual.poses["left::panda"].translation()
            total_replans += 1
        else:
            predicted_actions = None

        adapter.step_observations(obs)
        executed_action = adapter.step_action()
        if predicted_actions is not None:
            expected_action = predicted_actions[num_past]
            np.testing.assert_allclose(
                executed_action.poses["left::panda"].translation(),
                expected_action.poses["left::panda"].translation(),
                atol=1e-6,
            )
        executed_translations.append(executed_action.poses["left::panda"].translation())

        open_loop_counter = (open_loop_counter + 1) % open_loop_steps

    expected_replans = (total_steps - 1 + open_loop_steps - 1) // open_loop_steps
    assert total_replans == expected_replans
    assert len(processor.process_inputs_calls) == total_replans
    assert len(processor.normalizer.normalize_calls) == total_replans * len(action_fields)
    assert len(processor.normalizer.denormalize_calls) == total_replans * len(action_fields)

    recent_history = executed_translations[-num_past:]
    for idx, expected_translation in enumerate(recent_history):
        np.testing.assert_allclose(
            adapter.action_buffer[idx]["robot__action__poses__left::panda__xyz"], expected_translation, atol=1e-6
        )

    assert last_replan_translation is not None
    np.testing.assert_allclose(
        adapter.reference["robot__actual__poses__left::panda__xyz"], last_replan_translation, atol=1e-6
    )


def test_reset_initializes_buffers_and_reference(
    mock_robotics_processor, mock_data_config, field_mapping_file, create_multiarm_observation
):
    camera_names = _camera_basenames(mock_data_config.image_names)
    adapter = PolicyDataAdapter(
        robotics_processor=mock_robotics_processor,
        data_config=mock_data_config,
        field_mapping_path=field_mapping_file,
        image_names=mock_data_config.image_names,
        preprocessor_image_size=(mock_data_config.image_size, mock_data_config.image_size),
        num_past_timesteps=5,
        num_future_timesteps=2,
        image_indices=(-1, 0),
    )

    observation = create_multiarm_observation(0, camera_names=camera_names)
    adapter.reset(observation)

    expected_timesteps = adapter.num_past_timesteps + 1 + adapter.num_future_timesteps

    assert len(adapter.action_buffer) == expected_timesteps
    assert len(adapter.image_buffer) == 2
    expected_timesteps = adapter.num_past_timesteps + 1 + adapter.num_future_timesteps
    assert adapter.past_mask.shape == (1, expected_timesteps)

    for action_field in mock_data_config.action_fields:
        actual_field = relative_to_absolute_map(any_to_actual_map(action_field))
        assert actual_field in adapter.reference


def test_get_model_input_structure(
    mock_robotics_processor, mock_data_config, field_mapping_file, create_multiarm_observation
):
    camera_names = _camera_basenames(mock_data_config.image_names)
    adapter = PolicyDataAdapter(
        robotics_processor=mock_robotics_processor,
        data_config=mock_data_config,
        field_mapping_path=field_mapping_file,
        image_names=mock_data_config.image_names,
        preprocessor_image_size=(mock_data_config.image_size, mock_data_config.image_size),
        num_past_timesteps=5,
        num_future_timesteps=20,
        image_indices=(-5, 0),
    )

    observation = create_multiarm_observation(0, camera_names=camera_names)
    adapter.reset(observation)

    model_input = adapter.get_model_input(observation)

    expected_timesteps = adapter.num_past_timesteps + 1 + adapter.num_future_timesteps

    assert set(model_input.keys()) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "language_instruction",
        "lowdim",
        "actions",
        "past_mask",
    }
    assert model_input["input_ids"].shape == (1, 512)
    assert model_input["attention_mask"].shape == (1, 512)
    assert model_input["pixel_values"].shape == (1, 12, 3, 224, 224)
    assert model_input["actions"].shape == (1, expected_timesteps, adapter.action_dim)
    assert model_input["past_mask"].shape == (1, expected_timesteps)
    assert isinstance(model_input["language_instruction"], list)
    assert isinstance(model_input["lowdim"], dict)


def test_policy_data_adapter_proprioception_integration(field_mapping_file, create_multiarm_observation):
    """Verify proprioception buffers, metadata, and lowdim conversion."""

    num_past = 2
    num_future = 1
    image_names = ["cam0_t-1", "cam0_t0"]
    action_fields = [
        "robot__action__poses__left::panda__xyz_relative",
        "robot__action__poses__left::panda__rot_6d_relative",
        "robot__action__grippers__left::panda_hand",
    ]
    pose_groups = [
        {
            "name": "left_panda_action",
            "position_key": "robot__action__poses__left::panda__xyz",
            "rotation_key": "robot__action__poses__left::panda__rot_6d",
        }
    ]
    proprioception_fields = [
        "robot__actual__poses__left::panda__xyz",
        "robot__actual__poses__left::panda__rot_6d",
        "robot__actual__grippers__left::panda_hand",
    ]
    field_dims = {
        **{
            action_fields[0]: 3,
            action_fields[1]: 6,
            action_fields[2]: 1,
        },
        **{
            proprioception_fields[0]: 3,
            proprioception_fields[1]: 6,
            proprioception_fields[2]: 1,
        },
    }

    processor = FakeRoboticsProcessor(field_dims=field_dims, timestep_dim=num_past + 1 + num_future)
    augmentation = SimpleNamespace(image=SimpleNamespace(crop=SimpleNamespace(shape=(112, 112))))
    data_config = SimpleNamespace(
        action_fields=action_fields,
        image_names=image_names,
        augmentation=augmentation,
        proprioception_fields=proprioception_fields,
        pose_groups=pose_groups,
    )

    adapter = PolicyDataAdapter(
        robotics_processor=processor,
        data_config=data_config,
        field_mapping_path=field_mapping_file,
        image_names=image_names,
        preprocessor_image_size=(128, 128),
        num_past_timesteps=num_past,
        num_future_timesteps=num_future,
        image_indices=(-1, 0),
    )

    obs0 = create_multiarm_observation(step=0, camera_names=["cam0"])
    obs1 = create_multiarm_observation(step=1, camera_names=["cam0"])

    adapter.reset(obs0)
    processor.process_inputs_calls.clear()
    processor.normalizer.normalize_calls.clear()

    adapter.step_observations(obs1)
    model_input = adapter.get_model_input(obs1)

    assert "proprioception" in model_input
    proprioception_tensor = model_input["proprioception"]
    proprioception_dim = sum(field_dims[f] for f in proprioception_fields)
    assert proprioception_tensor.shape == (1, num_past + 1, proprioception_dim)

    batch, _ = processor.process_inputs_calls[-1]
    assert "metadata" in batch
    assert batch["metadata"][0]["anchor_relative_idx"] == num_past

    normalized_calls = {field: tensor for field, tensor in processor.normalizer.normalize_calls}
    for field in proprioception_fields:
        expected_stack = torch.tensor(
            np.stack(
                [entry[field] for entry in adapter.proprioception_buffer],
                axis=0,
            ).astype(np.float32)
        )
        if expected_stack.ndim == 1:
            expected_stack = expected_stack[:, None]
        torch.testing.assert_close(normalized_calls[field], expected_stack)


def test_stack_fields_absolute_fields(field_mapping_file):
    """Test _stack_fields with absolute (non-relative) fields only."""
    num_past = 1
    num_future = 2
    image_names = []
    action_fields = [
        "robot__action__grippers__left::panda_hand",
        "robot__action__grippers__right::panda_hand",
    ]
    field_dims = {
        "robot__action__grippers__left::panda_hand": 1,
        "robot__action__grippers__right::panda_hand": 1,
    }

    processor = FakeRoboticsProcessor(field_dims=field_dims, timestep_dim=num_past + 1 + num_future)
    augmentation = SimpleNamespace(image=SimpleNamespace(crop=SimpleNamespace(shape=(112, 112))))
    data_config = SimpleNamespace(
        action_fields=action_fields,
        image_names=image_names,
        augmentation=augmentation,
        proprioception_fields=[],
        pose_groups=[],
    )

    adapter = PolicyDataAdapter(
        robotics_processor=processor,
        data_config=data_config,
        field_mapping_path=field_mapping_file,
        image_names=image_names,
        preprocessor_image_size=(128, 128),
        num_past_timesteps=num_past,
        num_future_timesteps=num_future,
        image_indices=(-1, 0),
    )

    # Create buffer with known values
    buffer = [
        {
            "robot__action__grippers__left::panda_hand": np.array([0.5], dtype=np.float64),
            "robot__action__grippers__right::panda_hand": np.array([0.6], dtype=np.float64),
        },
        {
            "robot__action__grippers__left::panda_hand": np.array([0.7], dtype=np.float64),
            "robot__action__grippers__right::panda_hand": np.array([0.8], dtype=np.float64),
        },
        {
            "robot__action__grippers__left::panda_hand": np.array([0.9], dtype=np.float64),
            "robot__action__grippers__right::panda_hand": np.array([1.0], dtype=np.float64),
        },
    ]

    result = adapter._stack_fields(buffer, action_fields, relative_fields=[])

    # Check that all fields are present
    assert set(result.keys()) == set(action_fields)

    # Check shapes (should be [timesteps, dim])
    assert result["robot__action__grippers__left::panda_hand"].shape == (3, 1)
    assert result["robot__action__grippers__right::panda_hand"].shape == (3, 1)

    # Check values
    expected_left = torch.tensor([[0.5], [0.7], [0.9]], dtype=torch.float32)
    expected_right = torch.tensor([[0.6], [0.8], [1.0]], dtype=torch.float32)
    torch.testing.assert_close(result["robot__action__grippers__left::panda_hand"], expected_left)
    torch.testing.assert_close(result["robot__action__grippers__right::panda_hand"], expected_right)


def test_stack_fields_relative_pose_groups(field_mapping_file):
    """Test _stack_fields with relative pose groups (xyz + rot_6d)."""
    num_past = 1
    num_future = 1
    image_names = []
    action_fields = [
        "robot__action__poses__left::panda__xyz_relative",
        "robot__action__poses__left::panda__rot_6d_relative",
    ]
    pose_groups = [
        {
            "name": "left_panda_action",
            "position_key": "robot__action__poses__left::panda__xyz",
            "rotation_key": "robot__action__poses__left::panda__rot_6d",
        }
    ]
    field_dims = {
        action_fields[0]: 3,
        action_fields[1]: 6,
    }

    processor = FakeRoboticsProcessor(field_dims=field_dims, timestep_dim=num_past + 1 + num_future)
    augmentation = SimpleNamespace(image=SimpleNamespace(crop=SimpleNamespace(shape=(112, 112))))
    data_config = SimpleNamespace(
        action_fields=action_fields,
        image_names=image_names,
        augmentation=augmentation,
        proprioception_fields=[],
        pose_groups=pose_groups,
    )

    adapter = PolicyDataAdapter(
        robotics_processor=processor,
        data_config=data_config,
        field_mapping_path=field_mapping_file,
        image_names=image_names,
        preprocessor_image_size=(128, 128),
        num_past_timesteps=num_past,
        num_future_timesteps=num_future,
        image_indices=(-1, 0),
    )

    # Set reference position and rotation (identity)
    reference_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    reference_rot_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    adapter.reference = {
        "robot__actual__poses__left::panda__xyz": reference_xyz,
        "robot__actual__poses__left::panda__rot_6d": reference_rot_6d,
    }

    # Create buffer with absolute values
    # First timestep: at reference (should give zero relative)
    # Second timestep: translated by [0.1, 0.0, 0.0]
    # Third timestep: translated by [0.2, 0.0, 0.0]
    buffer = [
        {
            "robot__action__poses__left::panda__xyz": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "robot__action__poses__left::panda__rot_6d": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
        },
        {
            "robot__action__poses__left::panda__xyz": np.array([0.1, 0.0, 0.0], dtype=np.float64),
            "robot__action__poses__left::panda__rot_6d": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
        },
        {
            "robot__action__poses__left::panda__xyz": np.array([0.2, 0.0, 0.0], dtype=np.float64),
            "robot__action__poses__left::panda__rot_6d": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
        },
    ]

    result = adapter._stack_fields(buffer, action_fields, relative_fields=action_fields)

    # Check that all fields are present
    assert set(result.keys()) == set(action_fields)

    # Check shapes
    assert result["robot__action__poses__left::panda__xyz_relative"].shape == (3, 3)
    assert result["robot__action__poses__left::panda__rot_6d_relative"].shape == (3, 6)

    # Check relative xyz values (should match the translations relative to reference)
    expected_xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        result["robot__action__poses__left::panda__xyz_relative"],
        expected_xyz,
        atol=1e-5,
        rtol=1e-5,
    )

    # Check relative rot_6d values (should be identity since no rotation change)
    expected_rot_6d = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]] * 3,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        result["robot__action__poses__left::panda__rot_6d_relative"],
        expected_rot_6d,
        atol=1e-5,
        rtol=1e-5,
    )


def test_stack_fields_relative_joint_positions(field_mapping_file):
    """Test _stack_fields with relative joint positions."""
    num_past = 1
    num_future = 1
    image_names = []
    action_fields = [
        "robot__action__joint_position__left::panda_relative",
    ]
    field_dims = {
        action_fields[0]: 7,
    }

    processor = FakeRoboticsProcessor(field_dims=field_dims, timestep_dim=num_past + 1 + num_future)
    augmentation = SimpleNamespace(image=SimpleNamespace(crop=SimpleNamespace(shape=(112, 112))))
    data_config = SimpleNamespace(
        action_fields=action_fields,
        image_names=image_names,
        augmentation=augmentation,
        proprioception_fields=[],
        pose_groups=[],
    )

    adapter = PolicyDataAdapter(
        robotics_processor=processor,
        data_config=data_config,
        field_mapping_path=field_mapping_file,
        image_names=image_names,
        preprocessor_image_size=(128, 128),
        num_past_timesteps=num_past,
        num_future_timesteps=num_future,
        image_indices=(-1, 0),
    )

    # Set reference joint positions
    reference_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    adapter.reference = {
        "robot__actual__joint_position__left::panda": reference_joints,
    }

    # Create buffer with absolute joint positions
    buffer = [
        {
            "robot__action__joint_position__left::panda": np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64
            ),
        },
        {
            "robot__action__joint_position__left::panda": np.array(
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64
            ),
        },
    ]

    result = adapter._stack_fields(buffer, action_fields, relative_fields=action_fields)

    # Check that field is present
    assert "robot__action__joint_position__left::panda_relative" in result

    # Check shape
    assert result["robot__action__joint_position__left::panda_relative"].shape == (2, 7)

    # Check that values are wrapped to [-pi, pi]
    expected = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        result["robot__action__joint_position__left::panda_relative"],
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


def test_stack_fields_with_timestep_slice(field_mapping_file):
    """Test _stack_fields with timestep slicing."""
    num_past = 1
    num_future = 2
    image_names = []
    action_fields = ["robot__action__grippers__left::panda_hand"]
    field_dims = {"robot__action__grippers__left::panda_hand": 1}

    processor = FakeRoboticsProcessor(field_dims=field_dims, timestep_dim=num_past + 1 + num_future)
    augmentation = SimpleNamespace(image=SimpleNamespace(crop=SimpleNamespace(shape=(112, 112))))
    data_config = SimpleNamespace(
        action_fields=action_fields,
        image_names=image_names,
        augmentation=augmentation,
        proprioception_fields=[],
        pose_groups=[],
    )

    adapter = PolicyDataAdapter(
        robotics_processor=processor,
        data_config=data_config,
        field_mapping_path=field_mapping_file,
        image_names=image_names,
        preprocessor_image_size=(128, 128),
        num_past_timesteps=num_past,
        num_future_timesteps=num_future,
        image_indices=(-1, 0),
    )

    # Create buffer with 5 timesteps
    buffer = [
        {"robot__action__grippers__left::panda_hand": np.array([0.1], dtype=np.float64)},
        {"robot__action__grippers__left::panda_hand": np.array([0.2], dtype=np.float64)},
        {"robot__action__grippers__left::panda_hand": np.array([0.3], dtype=np.float64)},
        {"robot__action__grippers__left::panda_hand": np.array([0.4], dtype=np.float64)},
        {"robot__action__grippers__left::panda_hand": np.array([0.5], dtype=np.float64)},
    ]

    # Slice to get only middle 3 timesteps
    result = adapter._stack_fields(buffer, action_fields, relative_fields=[], timesteps_slice=slice(1, 4))

    assert result["robot__action__grippers__left::panda_hand"].shape == (3, 1)
    expected = torch.tensor([[0.2], [0.3], [0.4]], dtype=torch.float32)
    torch.testing.assert_close(result["robot__action__grippers__left::panda_hand"], expected)


def test_stack_fields_empty_buffer(field_mapping_file):
    """Test _stack_fields with empty buffer."""
    num_past = 1
    num_future = 1
    image_names = []
    action_fields = ["robot__action__grippers__left::panda_hand"]
    field_dims = {"robot__action__grippers__left::panda_hand": 1}

    processor = FakeRoboticsProcessor(field_dims=field_dims, timestep_dim=num_past + 1 + num_future)
    augmentation = SimpleNamespace(image=SimpleNamespace(crop=SimpleNamespace(shape=(112, 112))))
    data_config = SimpleNamespace(
        action_fields=action_fields,
        image_names=image_names,
        augmentation=augmentation,
        proprioception_fields=[],
        pose_groups=[],
    )

    adapter = PolicyDataAdapter(
        robotics_processor=processor,
        data_config=data_config,
        field_mapping_path=field_mapping_file,
        image_names=image_names,
        preprocessor_image_size=(128, 128),
        num_past_timesteps=num_past,
        num_future_timesteps=num_future,
        image_indices=(-1, 0),
    )

    result = adapter._stack_fields([], action_fields, relative_fields=[])
    assert result == {}


def test_stack_fields_mixed_absolute_and_relative(field_mapping_file):
    """Test _stack_fields with both absolute and relative fields together."""
    num_past = 1
    num_future = 1
    image_names = []
    action_fields = [
        "robot__action__poses__left::panda__xyz_relative",
        "robot__action__poses__left::panda__rot_6d_relative",
        "robot__action__grippers__left::panda_hand",  # Absolute field
    ]
    pose_groups = [
        {
            "name": "left_panda_action",
            "position_key": "robot__action__poses__left::panda__xyz",
            "rotation_key": "robot__action__poses__left::panda__rot_6d",
        }
    ]
    field_dims = {
        action_fields[0]: 3,
        action_fields[1]: 6,
        action_fields[2]: 1,
    }

    processor = FakeRoboticsProcessor(field_dims=field_dims, timestep_dim=num_past + 1 + num_future)
    augmentation = SimpleNamespace(image=SimpleNamespace(crop=SimpleNamespace(shape=(112, 112))))
    data_config = SimpleNamespace(
        action_fields=action_fields,
        image_names=image_names,
        augmentation=augmentation,
        proprioception_fields=[],
        pose_groups=pose_groups,
    )

    adapter = PolicyDataAdapter(
        robotics_processor=processor,
        data_config=data_config,
        field_mapping_path=field_mapping_file,
        image_names=image_names,
        preprocessor_image_size=(128, 128),
        num_past_timesteps=num_past,
        num_future_timesteps=num_future,
        image_indices=(-1, 0),
    )

    # Set reference
    adapter.reference = {
        "robot__actual__poses__left::panda__xyz": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "robot__actual__poses__left::panda__rot_6d": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    }

    # Create buffer
    buffer = [
        {
            "robot__action__poses__left::panda__xyz": np.array([0.1, 0.0, 0.0], dtype=np.float64),
            "robot__action__poses__left::panda__rot_6d": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
            "robot__action__grippers__left::panda_hand": np.array([0.5], dtype=np.float64),
        },
        {
            "robot__action__poses__left::panda__xyz": np.array([0.2, 0.0, 0.0], dtype=np.float64),
            "robot__action__poses__left::panda__rot_6d": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
            "robot__action__grippers__left::panda_hand": np.array([0.8], dtype=np.float64),
        },
    ]

    relative_fields = [
        "robot__action__poses__left::panda__xyz_relative",
        "robot__action__poses__left::panda__rot_6d_relative",
    ]

    result = adapter._stack_fields(buffer, action_fields, relative_fields)

    # Check all fields are present
    assert set(result.keys()) == set(action_fields)

    # Check relative pose fields
    assert result["robot__action__poses__left::panda__xyz_relative"].shape == (2, 3)
    assert result["robot__action__poses__left::panda__rot_6d_relative"].shape == (2, 6)

    # Check absolute gripper field
    assert result["robot__action__grippers__left::panda_hand"].shape == (2, 1)
    expected_gripper = torch.tensor([[0.5], [0.8]], dtype=torch.float32)
    torch.testing.assert_close(result["robot__action__grippers__left::panda_hand"], expected_gripper)


def test_update_action_generates_valid_poses(
    mock_robotics_processor, mock_data_config, field_mapping_file, create_multiarm_observation
):
    camera_names = _camera_basenames(mock_data_config.image_names)
    adapter = PolicyDataAdapter(
        robotics_processor=mock_robotics_processor,
        data_config=mock_data_config,
        field_mapping_path=field_mapping_file,
        image_names=mock_data_config.image_names,
        preprocessor_image_size=(mock_data_config.image_size, mock_data_config.image_size),
        num_past_timesteps=5,
        num_future_timesteps=2,
        image_indices=(-1, 0),
    )

    observation = create_multiarm_observation(0, camera_names=camera_names)
    adapter.reset(observation)

    # Use the initialized action buffer to create a plausible model output tensor
    lowdim = adapter.get_lowdim_for_processor()
    normalized_fields = [
        adapter.robotics_processor.normalizer.normalize_tensor(lowdim[field], field) for field in adapter.action_fields
    ]
    model_output = torch.cat(normalized_fields, dim=-1).unsqueeze(0)
    adapter.update_action(observation, model_output)

    expected_timesteps = adapter.num_past_timesteps + 1 + adapter.num_future_timesteps
    assert len(adapter.action_buffer) == expected_timesteps

    pose_actions = [adapter.action_mapping.create_pose_and_gripper(entry) for entry in adapter.action_buffer]

    first_action = pose_actions[0]
    assert isinstance(first_action, PosesAndGrippers)
    assert "left::panda" in first_action.poses
    left_pose = first_action.poses["left::panda"]
    assert isinstance(left_pose, RigidTransform)
    np.testing.assert_allclose(
        left_pose.translation(),
        observation.robot.actual.poses["left::panda"].translation(),
        atol=1e-6,
    )

    first_buffer_entry = adapter.action_buffer[0]
    assert isinstance(first_buffer_entry, dict)
    np.testing.assert_allclose(
        first_buffer_entry["robot__action__poses__left::panda__xyz"],
        observation.robot.actual.poses["left::panda"].translation(),
        atol=1e-6,
    )
