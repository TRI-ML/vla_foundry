"""
Robotics data utilities.

This module provides helper functions for working with robotics data,
including extraction of proprioception and action data based on configuration.
"""

from typing import Any

import numpy as np
import yaml
from tdigest_rs import TDigest


def any_to_actual_key(field: str) -> str:
    """Convert any field name to its 'actual' counterpart for field mapping lookup.
    Expects field name to be in format: robot__<desired/actual/action>__...
      - __ are used as separators for the different parts of the field name
      - <desired/actual/action> is the type of the data
      - ... is the rest of the field name separated by __
    """
    parts = field.split("__")
    if len(parts) > 2:
        return "__".join(parts[0:1] + ["actual"] + parts[2:])
    else:
        return None


def normalize(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def load_action_field_config(config_path: str) -> dict[str, list[Any]]:
    """Load action field configuration from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    return {
        "action_key_fields": data.get("action_key_fields", []),
        "action_index_fields": data.get("action_index_fields", []),
        "pose_groups": data.get("pose_groups", []),
    }


def rot_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to rotation matrix using Gram-Schmidt orthogonalization.

    Mathematical formulation:
    Given 6D input representing the first 2 rows of a rotation matrix:
        a1 = rot_6d[:3]  (first row)
        a2 = rot_6d[3:]  (second row)

    Apply Gram-Schmidt orthogonalization:
        b1 = normalize(a1)
        b2 = normalize(a2 - proj(a2, b1))  where proj(a2, b1) = (a2 · b1) * b1
        b3 = b1 × b2  (cross product)

    Result: R = [b1; b2; b3] forms an orthonormal rotation matrix (as rows)

    This ensures the output is a proper rotation matrix (orthogonal with det(R) = 1).

    Broadcasting behavior: Supports both single and batch inputs seamlessly.
    - Single input [6,] -> Single output [3, 3]
    - Batch input [N, 6] -> Batch output [N, 3, 3]

    Args:
        rot_6d: 6D rotation data of shape (6,) or (N, 6)

    Returns:
        rotation_matrix: Shape (3, 3) or (N, 3, 3) rotation matrices

    Note: This uses a row-based convention (not the column-based Zhou et al. 2019 standard).
    """
    # Handle single vector or batch of vectors
    original_shape = rot_6d.shape
    if rot_6d.ndim == 1:
        rot_6d = rot_6d.reshape(1, 6)

    a1, a2 = rot_6d[..., :3], rot_6d[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    rot_matrices = np.stack((b1, b2, b3), axis=-2)

    # Return original shape
    if len(original_shape) == 1:
        return rot_matrices[0]
    else:
        return rot_matrices.reshape(original_shape[:-1] + (3, 3))


def get_xyz(pose) -> np.ndarray:
    return np.array(pose.translation(), dtype=np.float64)


def get_rot_6d(pose) -> np.ndarray:
    return np.array(pose.rotation().matrix()[:2, :].flatten(), dtype=np.float64)


def matrix_to_rot_6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to 6D rotation representation.

    Mathematical formulation:
    From a 3x3 rotation matrix R = [r1; r2; r3] (rows),
    extract the first 2 rows and flatten:
        rot_6d = [r1, r2] = [r1[0], r1[1], r1[2], r2[0], r2[1], r2[2]]

    The third row r3 can be reconstructed as r3 = r1 × r2 due to the
    orthonormality property of rotation matrices.

    This representation is continuous and avoids singularities present in
    other rotation representations like Euler angles.

    Broadcasting behavior: Supports both single and batch inputs seamlessly.
    - Single input [3, 3] -> Single output [6,]
    - Batch input [N, 3, 3] -> Batch output [N, 6]

    Inverse of rot_6d_to_matrix().

    Args:
        rotation_matrix: Rotation matrix/matrices of shape (3, 3) or (N, 3, 3)

    Returns:
        rot_6d: 6D rotation data of shape (6,) or (N, 6)

    Note: This uses a row-based convention (not the column-based Zhou et al. 2019 standard).
    """
    batch_dim = rotation_matrix.shape[:-2]
    rot_6d = rotation_matrix[..., :2, :].copy().reshape(batch_dim + (6,))
    return rot_6d


def to_pose_matrix(xyz: np.ndarray, rot_6d: np.ndarray) -> np.ndarray:
    """Convert xyz and rot_6d to full 4x4 pose matrix/matrices.

    Mathematical formulation:
    For a single pose:
        T = [R  t]
            [0  1]
    where:
        - R is the 3x3 rotation matrix (from rot_6d)
        - t is the 3x1 translation vector (xyz)
        - T is the 4x4 homogeneous transformation matrix

    For batch processing, this operation is vectorized across the time dimension.

    Broadcasting behavior: NO BROADCASTING SUPPORTED.
    Both inputs must have matching dimensions:
    - Both single: xyz (3,) and rot_6d (6,) -> output (4, 4)
    - Both batch: xyz (T, 3) and rot_6d (T, 6) -> output (T, 4, 4)
    - Mixed dimensions will result in errors to prevent silent bugs.

    Note: rot_6d_to_matrix() may internally handle single->batch conversion,
    creating asymmetric behavior. For batch xyz + single rot_6d, the single
    rotation will be applied to all positions via rot_6d_to_matrix's internal
    broadcasting.

    Args:
        xyz: Position data of shape (3,) or (T, 3)
        rot_6d: Rotation data of shape (6,) or (T, 6)

    Returns:
        pose_matrix: Shape (4, 4) or (T, 4, 4) pose matrix/matrices

    Raises:
        ValueError: For dimension mismatches that cannot be naturally handled
    """
    # Handle single timestep case
    if xyz.ndim == 1:
        rot_matrix = rot_6d_to_matrix(rot_6d)
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rot_matrix
        pose_matrix[:3, 3] = xyz
        return pose_matrix

    # Batch processing
    T = xyz.shape[0]
    rot_matrices = rot_6d_to_matrix(rot_6d)  # Shape: (T, 3, 3)

    # Create batch of identity matrices
    pose_matrices = np.tile(np.eye(4), (T, 1, 1))  # Shape: (T, 4, 4)

    # Set rotation parts
    pose_matrices[:, :3, :3] = rot_matrices

    # Set translation parts
    pose_matrices[:, :3, 3] = xyz

    return pose_matrices


def invert_homogeneous_transform(T: np.ndarray) -> np.ndarray:
    """
    Invert one or more 4x4 rigid homogeneous transforms (SE3).

    Args:
        T: shape (4,4) or (...,4,4)

    Returns:
        T_inv: shape (4,4) or (...,4,4)
    """
    T = np.asarray(T)

    if T.shape[-2:] != (4, 4):
        raise ValueError(f"Expected shape (...,4,4), got {T.shape}")

    # Validate homogeneous bottom row
    if not np.allclose(T[..., 3, :], np.array([0, 0, 0, 1])):
        raise ValueError("Not a standard homogeneous transform (bottom row not [0,0,0,1])")

    R = T[..., :3, :3]  # (...,3,3)
    t = T[..., :3, 3]  # (...,3)

    T_inv = np.broadcast_to(np.eye(4, dtype=T.dtype), T.shape).copy()
    T_inv[..., :3, :3] = np.swapaxes(R, -1, -2)  # Rᵀ
    T_inv[..., :3, 3] = -np.einsum("...ij,...j->...i", T_inv[..., :3, :3], t)

    return T_inv


def calculate_relative_pose(pose_matrix: np.ndarray, reference_pose_matrix: np.ndarray) -> np.ndarray:
    """Calculate relative pose matrix/matrices given pose(s) and a reference pose.

    Mathematical formulation:
        T_relative = T_reference^(-1) @ T_current

    where:
        - T_reference is the 4x4 reference pose matrix (MUST be single pose)
        - T_current is the 4x4 current pose matrix (or batch of matrices)
        - T_relative represents the transformation from reference frame to current frame
        - @ denotes matrix multiplication

    This computes the pose of the current frame expressed in the reference frame's
    coordinate system.

    Broadcasting behavior: LIMITED BROADCASTING SUPPORTED.
    - Single pose vs single reference: (4, 4) vs (4, 4) -> (4, 4)
    - Batch poses vs single reference: (T, 4, 4) vs (4, 4) -> (T, 4, 4)
    - Batch reference poses are NOT supported and will cause errors

    The inverse of a homogeneous transformation matrix is:
        T^(-1) = [R^T  -R^T*t]
                 [0     1     ]
    where R^T is the rotation transpose and t is the translation.

    Args:
        pose_matrix: Shape (4, 4) or (T, 4, 4) current pose matrix/matrices
        reference_pose_matrix: Shape (4, 4) reference pose matrix (single pose only)

    Returns:
        relative_pose: Shape (4, 4) or (T, 4, 4) relative pose matrix/matrices

    Raises:
        ValueError: If reference_pose_matrix is not shape (4, 4)
    """
    # Validate that reference pose is a single (4, 4) matrix
    if reference_pose_matrix.shape != (4, 4):
        raise ValueError(f"reference_pose_matrix must be shape (4, 4), got {reference_pose_matrix.shape}")

    reference_pose_inv = np.linalg.inv(reference_pose_matrix)

    # Handle single pose case
    if pose_matrix.ndim == 2:
        return reference_pose_inv @ pose_matrix

    # Batch processing - use broadcasting: (4, 4) @ (T, 4, 4) -> (T, 4, 4)
    return reference_pose_inv @ pose_matrix


def apply_relative_pose(relative_pose_matrix: np.ndarray, reference_pose_matrix: np.ndarray) -> np.ndarray:
    """Apply a relative pose matrix to a reference pose matrix to get the absolute pose.

    Mathematical formulation:
        T_absolute = T_reference @ T_relative

    where:
        - T_reference is the 4x4 reference pose matrix (MUST be single pose)
        - T_relative is the 4x4 relative pose matrix (or batch of matrices)
        - T_absolute is the resulting absolute pose matrix
        - @ denotes matrix multiplication

    This operation composes two transformations: first the reference transformation,
    then the relative transformation. This is the inverse operation of
    calculate_relative_pose().

    Mathematically, this represents the composition of coordinate transformations:
    T_absolute = T_reference @ T_relative means "first apply T_relative, then T_reference"

    Broadcasting behavior: LIMITED BROADCASTING SUPPORTED.
    - Single relative vs single reference: (4, 4) vs (4, 4) -> (4, 4)
    - Batch relative vs single reference: (T, 4, 4) vs (4, 4) -> (T, 4, 4)
    - Batch reference poses are NOT supported and will cause errors

    Args:
        relative_pose_matrix: Shape (4, 4) or (T, 4, 4) relative pose matrix/matrices
        reference_pose_matrix: Shape (4, 4) reference pose matrix (single pose only)

    Returns:
        absolute_pose: Shape (4, 4) or (T, 4, 4) absolute pose matrix/matrices

    Raises:
        ValueError: If reference_pose_matrix is not shape (4, 4)
    """
    # Validate that reference pose is a single (4, 4) matrix
    if reference_pose_matrix.shape != (4, 4):
        raise ValueError(f"reference_pose_matrix must be shape (4, 4), got {reference_pose_matrix.shape}")

    # Handle single relative pose case
    if relative_pose_matrix.ndim == 2:
        return reference_pose_matrix @ relative_pose_matrix

    # Batch processing - use broadcasting: (4, 4) @ (T, 4, 4) -> (T, 4, 4)
    return reference_pose_matrix @ relative_pose_matrix


def pose_to_9d(pose_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract xyz and rot_6d components from a pose matrix.

    Mathematical formulation:
    From a 4x4 homogeneous transformation matrix:
        T = [R  t]
            [0  1]

    Extract:
        - xyz = t (translation vector from T[:3, 3])
        - rot_6d = first 2 rows of R flattened (rotation matrix from T[:3, :3])

    The 6D rotation representation stores the first 2 rows of the 3x3 rotation
    matrix R. The third row can be reconstructed via Gram-Schmidt orthogonalization
    since rotation matrices are orthonormal.

    Broadcasting behavior: Supports both single and batch inputs seamlessly.
    - Single input (4, 4) -> outputs (3,) and (6,)
    - Batch input (T, 4, 4) -> outputs (T, 3) and (T, 6)

    Args:
        pose_matrix: Shape (4, 4) or (T, 4, 4) pose matrix/matrices

    Returns:
        xyz: Shape (3,) or (T, 3) position data
        rot_6d: Shape (6,) or (T, 6) rotation data
    """
    if pose_matrix.ndim == 2:
        # Single pose matrix
        xyz = pose_matrix[:3, 3]
        rot_matrix = pose_matrix[:3, :3]
        rot_6d = matrix_to_rot_6d(rot_matrix)
        return xyz, rot_6d
    else:
        # Batch of pose matrices
        xyz = pose_matrix[:, :3, 3]  # Shape: (T, 3)
        rot_matrices = pose_matrix[:, :3, :3]  # Shape: (T, 3, 3)
        rot_6d = matrix_to_rot_6d(rot_matrices)  # Shape: (T, 6)
        return xyz, rot_6d


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert roll, pitch, yaw angles into a 3×3 rotation matrix.

    Args:
        roll: Rotation about the x-axis in radians.
        pitch: Rotation about the y-axis in radians.
        yaw: Rotation about the z-axis in radians.

    Returns:
        A (3, 3) array representing the rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ (Ry @ Rx)


def xyzrpy_to_T(pose: np.ndarray | list[float]) -> np.ndarray:
    """
    Convert [x, y, z, roll, pitch, yaw] vector(s) into homogeneous transform(s).

    Args:
        pose: Either a length-6 vector [x, y, z, roll, pitch, yaw] (angles in radians),
            or an array of shape (N, 6) where each row is one pose.

    Returns:
        An array of shape (N, 4, 4) with homogeneous transforms. If a single pose is provided,
        N = 1.
    """
    arr = np.asarray(pose, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 6:
            raise ValueError("Pose must be length 6 (x, y, z, roll, pitch, yaw)")
        arr = arr[None, :]
    elif arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("Pose must have shape (n, 6)")

    Ts: list[np.ndarray] = []
    for x, y, z, r, p, y_ in arr:
        T = np.eye(4)
        T[:3, :3] = rpy_to_R(r, p, y_)
        T[:3, 3] = [x, y, z]
        Ts.append(T)
    return np.stack(Ts, axis=0)


# Example usage:
if __name__ == "__main__":
    # Example with pose matrices and relative transformations
    np.random.seed(42)

    # Create sample poses using xyz positions and 6D rotations
    xyz_positions = np.array([[1.0, 2.0, 3.0], [1.5, 2.2, 3.1], [2.0, 2.5, 3.3], [2.2, 2.8, 3.5]])

    # Generate random 6D rotations by creating rotation matrices
    rot_6d_positions = []
    for _ in range(4):
        # Create a random rotation matrix using QR decomposition
        A = np.random.randn(3, 3)
        Q, R = np.linalg.qr(A)
        # Ensure proper rotation (det = 1)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        # Convert to 6D using the conversion function
        rot_6d = matrix_to_rot_6d(Q)
        rot_6d_positions.append(rot_6d)

    rot_6d_positions = np.array(rot_6d_positions)

    # Convert to pose matrices
    pose_matrices = to_pose_matrix(xyz_positions, rot_6d_positions)
    print("Original pose matrices shape:", pose_matrices.shape)

    # Use second pose as reference and calculate relative poses
    reference_idx = 1
    reference_pose = pose_matrices[reference_idx]
    relative_poses = calculate_relative_pose(pose_matrices, reference_pose)

    # Show original vs relative like the old examples
    print("Original xyz positions:")
    print(xyz_positions)
    print(f"\nRelative xyz to index {reference_idx}:")
    for i, rel_pose in enumerate(relative_poses):
        rel_xyz, _ = pose_to_9d(rel_pose)
        print(f"Position {i}: {rel_xyz}")

    print("\nOriginal 6D rotations:")
    print(rot_6d_positions)
    print(f"\nRelative 6D rotations to index {reference_idx}:")
    for i, rel_pose in enumerate(relative_poses):
        _, rel_rot_6d = pose_to_9d(rel_pose)
        print(f"Rotation {i}: {rel_rot_6d}")

    # Example of converting back to absolute poses
    recovered_poses = np.array([apply_relative_pose(rel_pose, reference_pose) for rel_pose in relative_poses])
    print("\nRecovered poses match original:", np.allclose(pose_matrices, recovered_poses))

    # Example with xyzrpy format
    xyzrpy_poses = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # Identity pose
            [1, 0, 0, np.pi / 4, 0, 0],  # Translation + rotation
        ]
    )
    pose_matrices_from_rpy = xyzrpy_to_T(xyzrpy_poses)
    print("\nPose matrices from RPY shape:", pose_matrices_from_rpy.shape)


def crop_sequence(
    data,
    anchor_idx,
    past_timesteps,
    future_timesteps,
):
    """
    Crop a sequence to specified past and future timesteps around an anchor point.

    Args:
        data: Array of shape [T, ...] where T is the total number of timesteps
        anchor_idx: Index of the anchor timestep in the original sequence
        past_timesteps: Number of past timesteps to keep (not including anchor)
        future_timesteps: Number of future timesteps to keep (including anchor)

    Returns:
        Cropped array of shape [past_timesteps + 1 +future_timesteps, ...]
    """
    # Calculate the range to extract
    assert anchor_idx >= past_timesteps
    assert data.shape[0] >= anchor_idx + future_timesteps + 1
    start_idx = anchor_idx - past_timesteps
    end_idx = anchor_idx + future_timesteps + 1

    return data[start_idx:end_idx]


def merge_percentiles_from_tdigest(states_list: list[dict[str, Any]], target_p: float) -> np.ndarray:
    """
    Merge percentiles by merging t-digest states and querying the merged digest.

    T-digest supports native merging of centroids, which provides accurate
    percentile estimates especially for tail quantiles.
    """
    valid_states = [s for s in states_list if s is not None and "digests" in s]
    if not valid_states:
        return None

    # Get shape from first valid state
    example_shape = tuple(valid_states[0].get("shape", []))
    if not example_shape:
        return None

    result = np.zeros(example_shape)

    # For each index position, merge all t-digests and query the percentile
    for idx in np.ndindex(example_shape):
        idx_str = str(idx)
        idx_list = list(idx)
        digests_to_merge = []
        buffer_samples = []

        for state in valid_states:
            # Handle compact serialization format
            if "indices" in state["digests"]:
                indices = state["digests"]["indices"]
                if idx_list in indices:
                    pos = indices.index(idx_list)
                    means = np.array(state["digests"]["means"][pos], dtype=np.float32)
                    weights = np.array(state["digests"]["weights"][pos], dtype=np.uint32)
                    compression = state.get("compression", 100)
                    digests_to_merge.append(TDigest.from_means_weights(means, weights, compression))
                elif "buffers" in state and isinstance(state["buffers"], dict) and "indices" in state["buffers"]:
                    # New sparse buffer format
                    b_indices = state["buffers"]["indices"]
                    if idx_list in b_indices:
                        pos = b_indices.index(idx_list)
                        buffer_samples.extend(state["buffers"]["data"][pos])
            else:
                # Legacy format handling (stringified tuples)
                if state.get("digests") and idx_str in state["digests"]:
                    digest_data = state["digests"][idx_str]
                    means = np.array(digest_data["means"], dtype=np.float32)
                    weights = np.array(digest_data["weights"], dtype=np.uint32)
                    compression = state.get("compression", 100)
                    digests_to_merge.append(TDigest.from_means_weights(means, weights, compression))
                elif state.get("buffer") is not None:
                    # Old monolithic buffer format
                    buffer = np.array(state["buffer"], dtype=np.float32)
                    counts = np.array(state["counts"], dtype=int)
                    cnt = counts[idx]
                    if cnt > 0:
                        selector = (slice(0, cnt),) + idx
                        buffer_samples.extend(buffer[selector].tolist())
                elif state.get("buffers") and idx_str in state["buffers"]:
                    # Intermediate sparse buffer format (dict[str, List])
                    buffer_samples.extend(state["buffers"][idx_str])

        # Create t-digest from buffer samples if any
        if buffer_samples:
            compression = valid_states[0].get("compression", 100)
            buffer_digest = TDigest.from_array(np.array(buffer_samples, dtype=np.float32), compression)
            digests_to_merge.append(buffer_digest)

        if digests_to_merge:
            # Merge all digests
            merged = digests_to_merge[0]
            for d in digests_to_merge[1:]:
                merged = merged.merge(d)
            result[idx] = merged.quantile(target_p)

    return result


def merge_statistics_single_field(tensor_stats: dict[str, list[Any]], stat_name: str) -> np.ndarray:
    """
    tensor_stats: {mean: [m1, m2, ... mn], std: [s1, s2, ... sn], ...}
    stat_name: mean, std, min, max, etc.
    """
    # Tensor sizes:
    # mean, min, max, etc. [num_datasets, action_dim]
    # mean_per_timestep, etc. [num_datasets, T, action_dim]
    # count [num_datasets, T]
    if stat_name == "mean":
        mean_per_timestep = merge_statistics_single_field(tensor_stats, "mean_per_timestep")
        counts = np.sum(tensor_stats["count"], axis=0)
        return np.average(mean_per_timestep, axis=0, weights=counts)
    elif stat_name == "mean_per_timestep":
        counts = np.broadcast_to(tensor_stats["count"][..., None], tensor_stats["mean_per_timestep"].shape)
        return np.average(tensor_stats["mean_per_timestep"], axis=0, weights=counts)
    elif stat_name == "std":
        # Use law of total variance: σ²_overall = E[σ²_t] + Var[μ_t]
        std_per_timestep = merge_statistics_single_field(tensor_stats, "std_per_timestep")
        variance_per_timestep = std_per_timestep**2
        mean_per_timestep = merge_statistics_single_field(tensor_stats, "mean_per_timestep")
        # Use weighted mean and variance based on counts per timestep
        counts_per_timestep = np.sum(tensor_stats["count"], axis=0)
        mean_variance = np.average(variance_per_timestep, axis=0, weights=counts_per_timestep)
        weighted_mean = np.average(mean_per_timestep, axis=0, weights=counts_per_timestep)
        variance_of_means = np.average((mean_per_timestep - weighted_mean) ** 2, axis=0, weights=counts_per_timestep)
        overall_variance = mean_variance + variance_of_means
        return np.sqrt(np.maximum(overall_variance, 0.0))

    elif stat_name == "std_per_timestep":
        # Use pooled variance formula to merge per-timestep standard deviations
        # σ²_pooled = [Σ((nᵢ-1)×σᵢ² + nᵢ×(μᵢ - μ_global)²)] / (n_total - 1)
        counts = np.array(tensor_stats["count"])[..., np.newaxis]  # [num_datasets, T, 1]
        total_counts = np.sum(counts, axis=0)  # [T, 1]
        variances = np.array(tensor_stats[stat_name]) ** 2

        pooled_mean_per_timestep = merge_statistics_single_field(tensor_stats, "mean_per_timestep")
        mean_diffs_squared = (tensor_stats["mean_per_timestep"] - pooled_mean_per_timestep[np.newaxis, :, :]) ** 2
        pooled_variance = np.sum((counts - 1) * variances + counts * mean_diffs_squared, axis=0) / np.maximum(
            total_counts - 1, 1
        )
        return np.sqrt(pooled_variance)

    elif stat_name in ["min", "min_per_timestep"]:
        return np.min(tensor_stats[stat_name], axis=0)
    elif stat_name in ["max", "max_per_timestep"]:
        return np.max(tensor_stats[stat_name], axis=0)
    elif stat_name in ["count"]:
        return np.sum(tensor_stats["count"], axis=0)
    elif stat_name in [
        "percentile_1",
        "percentile_2",
        "percentile_5",
        "percentile_95",
        "percentile_98",
        "percentile_99",
        "percentile_1_per_timestep",
        "percentile_2_per_timestep",
        "percentile_5_per_timestep",
        "percentile_95_per_timestep",
        "percentile_98_per_timestep",
        "percentile_99_per_timestep",
    ]:
        p_val_str_raw = stat_name.split("_")[1]
        p_val = float(p_val_str_raw) / 100.0
        is_per_timestep = "per_timestep" in stat_name

        state_key = "tdigest_state_per_timestep" if is_per_timestep else "tdigest_state"

        states = tensor_stats[state_key]
        return merge_percentiles_from_tdigest(states, p_val)

    elif stat_name in ["percentile_sample_count", "tdigest_state", "tdigest_state_per_timestep"]:
        return None  # We don't merge these directly; tdigest states are used for percentiles.
    else:
        raise ValueError(f"Invalid stat name: {stat_name}")


def merge_statistics(statistics: list[dict[str, Any]]) -> dict[str, Any]:
    """
    `statistics` is a list of dictionaries. Each item on the list represents a different dataset.
    Keys are tensor names. Values are dictionaries with keys mean, std, min, max, etc.

    Merging is done as follows:
    - mean - We can calculate this exactly
    - std - We can calculate this exactly (pooled variance formula)
    - min, max - We can calculate this exactly
    - percentiles - We take a weighted average of the percentiles weighted by the counts
    - count - We sum the counts
    """
    tensor_names, stat_names = set(), set()
    for dataset_statistics in statistics:
        tensor_names.update(dataset_statistics.keys())
        for tensor_name in dataset_statistics:
            stat_names.update(dataset_statistics[tensor_name].keys())

    # Create batched = {
    # robot:left:xyz: {mean: np.array([m1, m2, ... mn]), std: np.array([s1, s2, ... sn]), ...},
    # robot:right:xyz: {mean: np.array([m1, m2, ... mn]), std: np.array([s1, s2, ... sn]), ...},
    # ...
    # }
    batched_stats = {tensor_name: {s: [] for s in stat_names} for tensor_name in tensor_names}
    for tensor_name in tensor_names:
        for stat_name in stat_names:
            for dataset_statistics in statistics:
                val = dataset_statistics[tensor_name].get(stat_name) if tensor_name in dataset_statistics else None
                batched_stats[tensor_name][stat_name].append(val)

            has_none = any(v is None for v in batched_stats[tensor_name][stat_name])
            if has_none:
                pass
            elif stat_name in [
                "psquared_state",
                "psquared_state_per_timestep",
                "tdigest_state",
                "tdigest_state_per_timestep",
                "percentile_sample_count",
            ]:
                batched_stats[tensor_name][stat_name] = np.array(batched_stats[tensor_name][stat_name], dtype=object)
            else:
                try:
                    batched_stats[tensor_name][stat_name] = np.array(batched_stats[tensor_name][stat_name])
                except (ValueError, TypeError) as err:
                    raise ValueError(
                        f"Cannot merge statistics for '{tensor_name}.{stat_name}': "
                        f"inconsistent shapes across datasets (e.g. different numbers of timesteps)."
                    ) from err

    merged_stats = {}
    for tensor_name in batched_stats:
        merged_stats[tensor_name] = {}
        for stat_name in batched_stats[tensor_name]:
            merged_stats[tensor_name][stat_name] = merge_statistics_single_field(batched_stats[tensor_name], stat_name)
            if merged_stats[tensor_name][stat_name] is not None:
                merged_stats[tensor_name][stat_name] = (
                    merged_stats[tensor_name][stat_name].tolist()
                    if not isinstance(merged_stats[tensor_name][stat_name], list)
                    else merged_stats[tensor_name][stat_name]
                )
    return merged_stats
