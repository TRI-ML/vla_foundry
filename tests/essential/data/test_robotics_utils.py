import numpy as np
import pytest
import yaml

from vla_foundry.data.robotics.utils import (
    any_to_actual_key,
    apply_relative_pose,
    calculate_relative_pose,
    crop_sequence,
    load_action_field_config,
    matrix_to_rot_6d,
    normalize,
    pose_to_9d,
    rot_6d_to_matrix,
    rpy_to_R,
    to_pose_matrix,
    xyzrpy_to_T,
)


def test_any_to_actual_key_converts_desired_to_actual():
    field = "robot__desired__end_effector__xyz"
    assert any_to_actual_key(field) == "robot__actual__end_effector__xyz"


def test_any_to_actual_key_returns_none_for_short_field():
    assert any_to_actual_key("robot") is None


def test_normalize_single_vector():
    vector = np.array([3.0, 4.0, 0.0])
    normalized = normalize(vector)
    expected = np.array([0.6, 0.8, 0.0])
    np.testing.assert_allclose(normalized, expected)


def test_normalize_batch_of_vectors():
    batch = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    normalized = normalize(batch)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(normalized, expected)


def test_rpy_to_R_identity():
    # Zero roll, pitch, yaw should yield identity matrix
    R = rpy_to_R(0.0, 0.0, 0.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-7)


def test_rpy_to_R_90deg_rotations():
    # 90 deg roll
    R_roll = rpy_to_R(np.pi / 2, 0, 0)
    expected_roll = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(R_roll, expected_roll, atol=1e-7)

    # 90 deg pitch
    R_pitch = rpy_to_R(0, np.pi / 2, 0)
    expected_pitch = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    np.testing.assert_allclose(R_pitch, expected_pitch, atol=1e-7)

    # 90 deg yaw
    R_yaw = rpy_to_R(0, 0, np.pi / 2)
    expected_yaw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(R_yaw, expected_yaw, atol=1e-7)


def test_rpy_to_R_is_valid_rotation_matrix():
    # A random set of angles
    r, p, y = 0.3, -0.7, 1.2
    R = rpy_to_R(r, p, y)
    # Orthonormal and right-handed (determinant +1)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-7)


def test_xyzrpy_to_T_single_pose():
    # [x, y, z, roll, pitch, yaw] all zeros
    pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    T = xyzrpy_to_T(pose)
    assert T.shape == (1, 4, 4)
    np.testing.assert_allclose(T[0, :3, :3], np.eye(3), atol=1e-7)
    np.testing.assert_allclose(T[0, :3, 3], np.array([1.0, 2.0, 3.0]), atol=1e-7)
    np.testing.assert_allclose(T[0, 3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7)


def test_xyzrpy_to_T_batch():
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # identity
            [1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2],  # yaw 90
        ]
    )
    T = xyzrpy_to_T(poses)
    assert T.shape == (2, 4, 4)

    # First transform is identity
    np.testing.assert_allclose(T[0], np.eye(4), atol=1e-7)

    # Second transform translation
    np.testing.assert_allclose(T[1, :3, 3], np.array([1.0, 2.0, 3.0]), atol=1e-7)

    # Second transform rotation equals Rz(90°)
    Rz_expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(T[1, :3, :3], Rz_expected, atol=1e-7)


def test_xyzrpy_to_T_invalid_length():
    # Should raise ValueError for wrong length
    with pytest.raises(ValueError):
        xyzrpy_to_T([1, 2, 3, 4, 5])


def test_xyzrpy_to_T_invalid_shape():
    # Should raise ValueError for wrong shape
    arr = np.ones((2, 5))
    with pytest.raises(ValueError):
        xyzrpy_to_T(arr)


def test_load_action_field_config_reads_yaml(tmp_path):
    config_path = tmp_path / "actions.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "action_key_fields": ["robot__desired__poses__right::panda__xyz"],
                "action_index_fields": ["robot__indices"],
            }
        )
    )

    config = load_action_field_config(str(config_path))
    assert config["action_key_fields"] == ["robot__desired__poses__right::panda__xyz"]
    assert config["action_index_fields"] == ["robot__indices"]


def test_rot_6d_to_matrix_identity():
    rot_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    matrix = rot_6d_to_matrix(rot_6d)
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-6)


def test_matrix_rot_6d_roundtrip():
    rotation_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rot_6d = matrix_to_rot_6d(rotation_matrix)
    reconstructed = rot_6d_to_matrix(rot_6d)
    np.testing.assert_allclose(reconstructed, rotation_matrix, atol=1e-6)


def test_pose_relative_roundtrip():
    # Test pose matrix relative transformations
    xyz_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    identity = np.eye(3)
    rotation_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rot_6d_sequence = np.stack([matrix_to_rot_6d(identity), matrix_to_rot_6d(rotation_matrix)])

    # Convert to pose matrices
    pose_matrices = to_pose_matrix(xyz_positions, rot_6d_sequence)
    reference_pose = pose_matrices[0]

    # Calculate relative poses
    relative_poses = calculate_relative_pose(pose_matrices, reference_pose)

    # Extract components and verify
    for i, rel_pose in enumerate(relative_poses):
        rel_xyz, rel_rot_6d = pose_to_9d(rel_pose)
        rel_rotation_matrix = rot_6d_to_matrix(rel_rot_6d)

        # First pose should be identity relative to itself
        if i == 0:
            np.testing.assert_allclose(rel_xyz, [0.0, 0.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(rel_rotation_matrix, identity, atol=1e-6)


def test_xyz_relative_with_pose_matrices():
    # Test xyz relative transformations using pose matrices
    xyz_sequence = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    # Use identity rotations for pure translation test
    rot_6d_sequence = np.tile(matrix_to_rot_6d(np.eye(3)), (3, 1))

    # Convert to pose matrices
    pose_matrices = to_pose_matrix(xyz_sequence, rot_6d_sequence)
    reference_pose = pose_matrices[1]  # Use middle pose as reference

    # Calculate relative poses
    relative_poses = calculate_relative_pose(pose_matrices, reference_pose)

    # Extract relative xyz positions
    relative_xyz = np.array([pose_to_9d(rel_pose)[0] for rel_pose in relative_poses])

    # Verify expected relative positions
    expected_relative = np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(relative_xyz, expected_relative, atol=1e-6)


def test_crop_sequence_extracts_expected_window():
    data = np.arange(10)
    cropped = crop_sequence(data, anchor_idx=5, past_timesteps=2, future_timesteps=3)
    np.testing.assert_array_equal(cropped, np.array([3, 4, 5, 6, 7, 8]))


def test_crop_sequence_invalid_anchor_asserts():
    with pytest.raises(AssertionError):
        crop_sequence(np.arange(5), anchor_idx=1, past_timesteps=2, future_timesteps=1)


class TestRotation6DProperties:
    """Test mathematical properties of 6D rotation representation."""

    def test_rotation_matrix_properties(self):
        """Test that 6D representation produces valid rotation matrices."""
        rot_6d = np.random.randn(6)
        rot_matrix = rot_6d_to_matrix(rot_6d)

        # Should be orthogonal: R @ R.T = I
        identity_check = rot_matrix @ rot_matrix.T
        np.testing.assert_allclose(identity_check, np.eye(3), atol=1e-6)

        # Should have determinant 1 (proper rotation, not reflection)
        det = np.linalg.det(rot_matrix)
        np.testing.assert_allclose(det, 1.0, atol=1e-6)

    def test_batch_rotation_matrix_properties(self):
        """Test rotation matrix properties for batches."""
        batch_size = 10
        rot_6d_batch = np.random.randn(batch_size, 6)
        rot_matrices = rot_6d_to_matrix(rot_6d_batch)

        for i in range(batch_size):
            # Each matrix should be orthogonal
            identity_check = rot_matrices[i] @ rot_matrices[i].T
            np.testing.assert_allclose(identity_check, np.eye(3), atol=1e-6)

            # Each matrix should have determinant 1
            det = np.linalg.det(rot_matrices[i])
            np.testing.assert_allclose(det, 1.0, atol=1e-6)

    def test_scale_invariance(self):
        """Test that scaling 6D representation doesn't change the resulting rotation matrix."""
        rot_6d_base = np.random.randn(6)
        scales = [0.1, 0.5, 1.0, 2.0, 10.0, 100.0]

        base_matrix = rot_6d_to_matrix(rot_6d_base)

        for scale in scales:
            rot_6d_scaled = rot_6d_base * scale
            scaled_matrix = rot_6d_to_matrix(rot_6d_scaled)

            # Scaled version should produce the same rotation matrix
            np.testing.assert_allclose(base_matrix, scaled_matrix, rtol=1e-6)

    def test_batch_scale_invariance(self):
        """Test scale invariance for batches."""
        batch_size = 5
        rot_6d_batch = np.random.randn(batch_size, 6)
        scales = np.random.uniform(0.1, 10.0, size=(batch_size, 1))

        base_matrices = rot_6d_to_matrix(rot_6d_batch)
        scaled_matrices = rot_6d_to_matrix(rot_6d_batch * scales)

        np.testing.assert_allclose(base_matrices, scaled_matrices, rtol=1e-6)

    def test_roundtrip_consistency(self):
        """Test that matrix→6D→matrix preserves the rotation."""
        # Start with a known rotation matrix
        angle = np.pi / 4
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        original_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        # Convert to 6D and back
        rot_6d = matrix_to_rot_6d(original_matrix)
        reconstructed_matrix = rot_6d_to_matrix(rot_6d)

        np.testing.assert_allclose(original_matrix, reconstructed_matrix, rtol=1e-10)

    def test_known_rotations(self):
        """Test with known geometric rotations."""
        # Identity rotation
        identity_6d = np.array([1, 0, 0, 0, 1, 0])
        identity_matrix = rot_6d_to_matrix(identity_6d)
        np.testing.assert_allclose(identity_matrix, np.eye(3), atol=1e-10)

        # 90-degree Z rotation (after normalization)
        z_rot_6d = np.array([0, 1, 0, -1, 0, 0])  # Will be normalized
        z_rot_matrix = rot_6d_to_matrix(z_rot_6d)
        expected_z_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # Corrected direction
        np.testing.assert_allclose(z_rot_matrix, expected_z_rot, atol=1e-10)


class TestPoseMatrixProperties:
    """Test mathematical properties of pose matrix operations."""

    def test_pose_matrix_structure(self):
        """Test that pose matrices have the correct homogeneous structure."""
        xyz = np.array([1.0, 2.0, 3.0])
        rot_6d = np.random.randn(6)

        pose_matrix = to_pose_matrix(xyz, rot_6d)

        # Check structure: [[R, t], [0, 1]]
        assert pose_matrix.shape == (4, 4)
        np.testing.assert_allclose(pose_matrix[3, :3], np.zeros(3), atol=1e-10)
        np.testing.assert_allclose(pose_matrix[3, 3], 1.0, atol=1e-10)
        np.testing.assert_allclose(pose_matrix[:3, 3], xyz, rtol=1e-10)

    def test_pose_decomposition_accuracy(self):
        """Test that pose decomposition exactly recovers position."""
        batch_size = 5
        xyz_original = np.random.randn(batch_size, 3) * 10.0
        rot_6d_original = np.random.randn(batch_size, 6)

        pose_matrices = to_pose_matrix(xyz_original, rot_6d_original)
        xyz_recovered, rot_6d_recovered = pose_to_9d(pose_matrices)

        # Position should be exactly preserved
        np.testing.assert_allclose(xyz_original, xyz_recovered, rtol=1e-12)

        # Rotation should be consistent
        rot_matrices_original = rot_6d_to_matrix(rot_6d_original)
        rot_matrices_recovered = rot_6d_to_matrix(rot_6d_recovered)
        np.testing.assert_allclose(rot_matrices_original, rot_matrices_recovered, rtol=1e-10)

    def test_known_poses(self):
        """Test with known geometric poses."""
        # Identity pose
        xyz_identity = np.array([0, 0, 0])
        rot_6d_identity = np.array([1, 0, 0, 0, 1, 0])
        pose_identity = to_pose_matrix(xyz_identity, rot_6d_identity)
        np.testing.assert_allclose(pose_identity, np.eye(4), atol=1e-10)

        # Pure translation
        xyz_translation = np.array([5, -3, 2])
        pose_translation = to_pose_matrix(xyz_translation, rot_6d_identity)
        expected_translation = np.eye(4)
        expected_translation[:3, 3] = xyz_translation
        np.testing.assert_allclose(pose_translation, expected_translation, atol=1e-10)


class TestSE3GroupProperties:
    """Test SE(3) group properties of pose transformations."""

    def test_identity_relative_pose(self):
        """Test that relative pose of identical poses is identity."""
        xyz = np.array([1.0, 2.0, 3.0])
        rot_6d = np.random.randn(6)
        pose_matrix = to_pose_matrix(xyz, rot_6d)

        # Relative pose of pose to itself should be identity
        relative_pose = calculate_relative_pose(pose_matrix, pose_matrix)
        np.testing.assert_allclose(relative_pose, np.eye(4), atol=1e-12)

    def test_inverse_property(self):
        """Test that T * T^(-1) = I."""
        xyz_current = np.array([1.5, -2.3, 4.1])
        rot_6d_current = np.random.randn(6)
        xyz_reference = np.array([0.5, 1.0, -1.5])
        rot_6d_reference = np.random.randn(6)

        pose_current = to_pose_matrix(xyz_current, rot_6d_current)
        pose_reference = to_pose_matrix(xyz_reference, rot_6d_reference)

        # Forward and back should give identity
        relative_pose = calculate_relative_pose(pose_current, pose_reference)
        reconstructed_pose = apply_relative_pose(relative_pose, pose_reference)

        np.testing.assert_allclose(pose_current, reconstructed_pose, rtol=1e-12)

    def test_composition_property(self):
        """Test that relative pose composition works correctly."""
        # Create three poses: A, B, C
        xyz_a = np.array([1, 0, 0])
        xyz_b = np.array([0, 1, 0])
        xyz_c = np.array([0, 0, 1])
        rot_6d_identity = np.array([1, 0, 0, 0, 1, 0])

        pose_a = to_pose_matrix(xyz_a, rot_6d_identity)
        pose_b = to_pose_matrix(xyz_b, rot_6d_identity)
        pose_c = to_pose_matrix(xyz_c, rot_6d_identity)

        # Test: relative_pose(C, A) = relative_pose(C, B) * relative_pose(B, A)
        rel_ca_direct = calculate_relative_pose(pose_c, pose_a)
        rel_cb = calculate_relative_pose(pose_c, pose_b)
        rel_ba = calculate_relative_pose(pose_b, pose_a)

        # Apply composition: first B→A, then C→B
        pose_a_via_b = apply_relative_pose(rel_ba, pose_a)  # Should equal pose_b
        pose_c_via_composition = apply_relative_pose(rel_cb, pose_a_via_b)

        # Verify the composition works correctly
        np.testing.assert_allclose(pose_b, pose_a_via_b, rtol=1e-12)
        np.testing.assert_allclose(pose_c, pose_c_via_composition, rtol=1e-12)

        # Also verify that the direct transformation from A to C gives the same result
        pose_c_via_direct = apply_relative_pose(rel_ca_direct, pose_a)
        np.testing.assert_allclose(pose_c, pose_c_via_direct, rtol=1e-12)

    def test_point_transformation_consistency(self):
        """Test that relative poses correctly transform points between coordinate frames."""
        # Create two poses
        xyz_ref = np.array([2, 3, 1])
        xyz_current = np.array([5, -1, 2])
        rot_6d_identity = np.array([1, 0, 0, 0, 1, 0])  # No rotation for simplicity

        pose_ref = to_pose_matrix(xyz_ref, rot_6d_identity)
        pose_current = to_pose_matrix(xyz_current, rot_6d_identity)

        # Calculate relative pose
        relative_pose = calculate_relative_pose(pose_current, pose_ref)

        # Point in reference frame
        point_ref = np.array([1, 1, 1, 1])  # Homogeneous coordinates

        # Transform point: ref → current
        point_current_expected = pose_current @ np.linalg.inv(pose_ref) @ point_ref
        point_current_via_relative = relative_pose @ point_ref

        np.testing.assert_allclose(point_current_expected[:3], point_current_via_relative[:3], rtol=1e-12)

    def test_batch_inverse_property(self):
        """Test inverse property for batches."""
        batch_size = 5
        xyz_sequence = np.random.randn(batch_size, 3) * 2.0
        rot_6d_sequence = np.random.randn(batch_size, 6)
        xyz_reference = np.random.randn(3)
        rot_6d_reference = np.random.randn(6)

        pose_matrices = to_pose_matrix(xyz_sequence, rot_6d_sequence)
        pose_reference = to_pose_matrix(xyz_reference, rot_6d_reference)

        # Forward and back transformation for each pose
        for i in range(batch_size):
            relative_pose = calculate_relative_pose(pose_matrices[i], pose_reference)
            reconstructed_pose = apply_relative_pose(relative_pose, pose_reference)
            np.testing.assert_allclose(pose_matrices[i], reconstructed_pose, rtol=1e-12)


class TestGeometricConsistency:
    """Test geometric consistency with known transformations."""

    def test_known_relative_transformations(self):
        """Test relative poses with known geometric relationships."""
        # Identity pose
        rot_6d_identity = np.array([1, 0, 0, 0, 1, 0])
        pose_origin = to_pose_matrix(np.array([0, 0, 0]), rot_6d_identity)

        # Pose translated by [1, 2, 3]
        pose_translated = to_pose_matrix(np.array([1, 2, 3]), rot_6d_identity)

        # Relative pose should be pure translation
        relative_pose = calculate_relative_pose(pose_translated, pose_origin)
        expected_relative = np.eye(4)
        expected_relative[:3, 3] = [1, 2, 3]

        np.testing.assert_allclose(relative_pose, expected_relative, atol=1e-12)

    def test_rotation_composition(self):
        """Test that rotation compositions work correctly."""
        # 90-degree rotations around Z-axis
        cos_45, sin_45 = np.cos(np.pi / 4), np.sin(np.pi / 4)

        # Two 45-degree rotations should equal one 90-degree rotation
        rot_45_6d = np.array([cos_45, sin_45, 0, -sin_45, cos_45, 0])
        pose_45_1 = to_pose_matrix(np.array([0, 0, 0]), rot_45_6d)
        pose_45_2 = to_pose_matrix(np.array([1, 0, 0]), rot_45_6d)  # Rotated and translated

        # Apply first rotation, then second
        relative_pose = calculate_relative_pose(pose_45_2, pose_45_1)
        final_pose = apply_relative_pose(relative_pose, pose_45_1)

        np.testing.assert_allclose(pose_45_2, final_pose, rtol=1e-12, atol=1e-15)  # Handle tiny numerical errors

    def test_coordinate_frame_consistency(self):
        """Test that coordinate frame transformations preserve geometric relationships."""
        # Create a triangle of poses
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        rot_6d_identity = np.array([1, 0, 0, 0, 1, 0])

        poses = [to_pose_matrix(v, rot_6d_identity) for v in vertices]

        # Use first pose as reference
        relative_poses = [calculate_relative_pose(pose, poses[0]) for pose in poses[1:]]

        # Reconstruct poses
        reconstructed_poses = [apply_relative_pose(rel_pose, poses[0]) for rel_pose in relative_poses]

        # Should preserve the original triangle
        for i, reconstructed in enumerate(reconstructed_poses):
            np.testing.assert_allclose(poses[i + 1], reconstructed, rtol=1e-12)


class TestNumericalRobustness:
    """Test numerical stability and robustness."""

    def test_extreme_scales(self):
        """Test with very small and very large scales."""
        base_rot_6d = np.array([1, 0.5, 0.2, -0.3, 0.8, 0.1])
        scales = [1e-10, 1e-5, 1e5, 1e10]

        base_matrix = rot_6d_to_matrix(base_rot_6d)

        for scale in scales:
            scaled_rot_6d = base_rot_6d * scale
            scaled_matrix = rot_6d_to_matrix(scaled_rot_6d)

            # Should produce same rotation regardless of scale
            np.testing.assert_allclose(base_matrix, scaled_matrix, rtol=1e-6)

    def test_near_degenerate_cases(self):
        """Test with nearly degenerate 6D representations."""
        # Nearly parallel first two columns (should be orthogonalized)
        rot_6d_near_parallel = np.array([1, 0, 0, 1.001, 0.001, 0])
        rot_matrix = rot_6d_to_matrix(rot_6d_near_parallel)

        # Should still produce valid rotation matrix
        identity_check = rot_matrix @ rot_matrix.T
        np.testing.assert_allclose(identity_check, np.eye(3), atol=1e-6)
        det = np.linalg.det(rot_matrix)
        np.testing.assert_allclose(det, 1.0, atol=1e-6)

    def test_precision_preservation(self):
        """Test that operations preserve reasonable precision."""
        # High precision input
        xyz = np.array([1.23456789012345, -2.34567890123456, 3.45678901234567])
        rot_6d = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        pose_matrix = to_pose_matrix(xyz, rot_6d)
        xyz_recovered, rot_6d_recovered = pose_to_9d(pose_matrix)

        # Position should be preserved to high precision
        np.testing.assert_allclose(xyz, xyz_recovered, rtol=1e-14)

    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_batch_consistency(self, batch_size):
        """Test that batch operations give same results as individual operations."""
        xyz_sequence = np.random.randn(batch_size, 3)
        rot_6d_sequence = np.random.randn(batch_size, 6)

        # Batch operation
        pose_matrices_batch = to_pose_matrix(xyz_sequence, rot_6d_sequence)

        # Individual operations
        pose_matrices_individual = np.array(
            [to_pose_matrix(xyz_sequence[i], rot_6d_sequence[i]) for i in range(batch_size)]
        )

        np.testing.assert_allclose(pose_matrices_batch, pose_matrices_individual, rtol=1e-15)


class TestBatchDimensionIntegration:
    """Test integration issues with batch dimension mismatches and strict validation."""

    def test_single_vs_batch_xyz_rot_mismatch_raises_error(self):
        """Test that mismatched batch dimensions raise clear errors."""
        # Single xyz, batch rot_6d - should raise error
        xyz_single = np.array([1.0, 2.0, 3.0])  # (3,)
        rot_6d_batch = np.random.randn(5, 6)  # (5, 6)

        with pytest.raises(ValueError, match="could not broadcast input array"):
            to_pose_matrix(xyz_single, rot_6d_batch)

    def test_batch_vs_single_xyz_rot_mismatch_behavior(self):
        """Test that batch xyz + single rot_6d works due to rot_6d_to_matrix internal broadcasting."""
        # Batch xyz, single rot_6d - this actually works because rot_6d_to_matrix handles it
        xyz_batch = np.random.randn(4, 3)  # (4, 3)
        rot_6d_single = np.random.randn(6)  # (6,)

        # This works because rot_6d_to_matrix broadcasts the single rot_6d internally
        pose_matrices = to_pose_matrix(xyz_batch, rot_6d_single)
        assert pose_matrices.shape == (4, 4, 4)

        # All poses should have the same rotation
        expected_rot_matrix = rot_6d_to_matrix(rot_6d_single)
        for i in range(4):
            actual_rot_matrix = pose_matrices[i, :3, :3]
            np.testing.assert_allclose(actual_rot_matrix, expected_rot_matrix, rtol=1e-12)

    def test_incompatible_batch_sizes_raise_error(self):
        """Test that incompatible batch sizes raise clear errors."""
        xyz_batch_3 = np.random.randn(3, 3)  # (3, 3)
        rot_6d_batch_5 = np.random.randn(5, 6)  # (5, 6)

        # These batch sizes are incompatible
        with pytest.raises(ValueError, match="could not broadcast input array"):
            to_pose_matrix(xyz_batch_3, rot_6d_batch_5)

    def test_matching_batch_dimensions_work(self):
        """Test that matching batch dimensions work correctly."""
        batch_size = 4
        xyz_batch = np.random.randn(batch_size, 3)
        rot_6d_batch = np.random.randn(batch_size, 6)

        # Should work without error
        pose_matrices = to_pose_matrix(xyz_batch, rot_6d_batch)
        assert pose_matrices.shape == (batch_size, 4, 4)

        # Test that each pose is constructed correctly
        for i in range(batch_size):
            individual_pose = to_pose_matrix(xyz_batch[i], rot_6d_batch[i])
            np.testing.assert_allclose(pose_matrices[i], individual_pose, rtol=1e-12)

    def test_relative_pose_batch_reference_mismatch(self):
        """Test relative pose calculation with batch vs single reference."""
        # Batch of poses
        xyz_batch = np.random.randn(4, 3)
        rot_6d_batch = np.random.randn(4, 6)
        pose_batch = to_pose_matrix(xyz_batch, rot_6d_batch)

        # Single reference pose
        xyz_ref = np.random.randn(3)
        rot_6d_ref = np.random.randn(6)
        pose_ref = to_pose_matrix(xyz_ref, rot_6d_ref)

        # Should work with broadcasting
        relative_poses = calculate_relative_pose(pose_batch, pose_ref)
        assert relative_poses.shape == (4, 4, 4)

    def test_calculate_relative_pose_rejects_batch_reference(self):
        """Test that calculate_relative_pose rejects batch reference poses."""
        # Create test data
        pose_single = np.eye(4)
        pose_batch = np.tile(np.eye(4), (3, 1, 1))
        reference_single = np.eye(4)
        reference_batch = np.tile(np.eye(4), (2, 1, 1))  # Batch reference (invalid)

        # Single vs single should work
        result = calculate_relative_pose(pose_single, reference_single)
        assert result.shape == (4, 4)

        # Batch vs single should work
        result = calculate_relative_pose(pose_batch, reference_single)
        assert result.shape == (3, 4, 4)

        # Single vs batch should fail
        with pytest.raises(ValueError, match="reference_pose_matrix must be shape \\(4, 4\\)"):
            calculate_relative_pose(pose_single, reference_batch)

        # Batch vs batch should fail
        with pytest.raises(ValueError, match="reference_pose_matrix must be shape \\(4, 4\\)"):
            calculate_relative_pose(pose_batch, reference_batch)

    def test_apply_relative_pose_rejects_batch_reference(self):
        """Test that apply_relative_pose rejects batch reference poses but supports batch relative poses."""
        relative_single = np.eye(4)
        reference_single = np.eye(4)
        relative_batch = np.tile(np.eye(4), (3, 1, 1))
        reference_batch = np.tile(np.eye(4), (2, 1, 1))

        # Single vs single should work
        result = apply_relative_pose(relative_single, reference_single)
        assert result.shape == (4, 4)

        # Batch relative vs single reference should work
        result = apply_relative_pose(relative_batch, reference_single)
        assert result.shape == (3, 4, 4)

        # Single relative vs batch reference should fail
        with pytest.raises(ValueError, match="reference_pose_matrix must be shape \\(4, 4\\)"):
            apply_relative_pose(relative_single, reference_batch)

        # Batch relative vs batch reference should fail
        with pytest.raises(ValueError, match="reference_pose_matrix must be shape \\(4, 4\\)"):
            apply_relative_pose(relative_batch, reference_batch)

    def test_relative_pose_roundtrip_consistency_batch(self):
        """Test that calculate_relative_pose and apply_relative_pose are mathematical inverses for batch data."""
        # Create batch of random poses
        np.random.seed(42)
        batch_size = 4
        xyz_batch = np.random.randn(batch_size, 3)
        rot_6d_batch = np.random.randn(batch_size, 6)
        original_poses = to_pose_matrix(xyz_batch, rot_6d_batch)

        # Single reference pose
        xyz_ref = np.random.randn(3)
        rot_6d_ref = np.random.randn(6)
        reference_pose = to_pose_matrix(xyz_ref, rot_6d_ref)

        # Calculate relative poses
        relative_poses = calculate_relative_pose(original_poses, reference_pose)
        assert relative_poses.shape == (batch_size, 4, 4)

        # Apply relative poses back to get original poses
        recovered_poses = apply_relative_pose(relative_poses, reference_pose)
        assert recovered_poses.shape == (batch_size, 4, 4)

        # Should recover original poses exactly
        np.testing.assert_allclose(original_poses, recovered_poses, rtol=1e-12)

        # Test individual calculation matches batch
        for i in range(batch_size):
            individual_relative = calculate_relative_pose(original_poses[i], reference_pose)
            np.testing.assert_allclose(relative_poses[i], individual_relative, rtol=1e-12)

    def test_calculate_relative_pose_batch_reference_error(self):
        """Test that batch reference poses are not supported."""
        # Single pose
        xyz_single = np.random.randn(3)
        rot_6d_single = np.random.randn(6)
        pose_single = to_pose_matrix(xyz_single, rot_6d_single)

        # Batch reference poses - should not be supported
        xyz_batch_ref = np.random.randn(3, 3)
        rot_6d_batch_ref = np.random.randn(3, 6)
        pose_batch_ref = to_pose_matrix(xyz_batch_ref, rot_6d_batch_ref)

        with pytest.raises((ValueError, IndexError)):
            calculate_relative_pose(pose_single, pose_batch_ref)

    def test_pose_components_batch_dimensions(self):
        """Test pose_to_9d with various batch dimensions."""
        # Single pose
        xyz_single = np.array([1, 2, 3])
        rot_6d_single = np.random.randn(6)
        pose_single = to_pose_matrix(xyz_single, rot_6d_single)

        xyz_out, rot_6d_out = pose_to_9d(pose_single)
        assert xyz_out.shape == (3,)
        assert rot_6d_out.shape == (6,)

        # Batch poses
        xyz_batch = np.random.randn(7, 3)
        rot_6d_batch = np.random.randn(7, 6)
        pose_batch = to_pose_matrix(xyz_batch, rot_6d_batch)

        xyz_out_batch, rot_6d_out_batch = pose_to_9d(pose_batch)
        assert xyz_out_batch.shape == (7, 3)
        assert rot_6d_out_batch.shape == (7, 6)

    def test_rotation_matrix_batch_consistency(self):
        """Test that rotation matrix operations handle batching consistently."""
        batch_size = 6

        # Test single input
        rot_6d_single = np.random.randn(6)
        matrix_single = rot_6d_to_matrix(rot_6d_single)
        assert matrix_single.shape == (3, 3)

        # Test batch input
        rot_6d_batch = np.random.randn(batch_size, 6)
        matrix_batch = rot_6d_to_matrix(rot_6d_batch)
        assert matrix_batch.shape == (batch_size, 3, 3)

        # Test conversion back
        rot_6d_recovered_single = matrix_to_rot_6d(matrix_single)
        rot_6d_recovered_batch = matrix_to_rot_6d(matrix_batch)
        assert rot_6d_recovered_single.shape == (6,)
        assert rot_6d_recovered_batch.shape == (
            batch_size,
            6,
        )

    def test_unexpected_broadcasting_scenarios(self):
        """Test edge cases where numpy broadcasting might behave unexpectedly."""
        # Test with shape (1, 3) vs (3,) - both should work but might have subtle differences
        xyz_1x3 = np.random.randn(1, 3)  # Shape (1, 3)
        xyz_3 = xyz_1x3[0]  # Shape (3,)
        rot_6d = np.random.randn(6)

        pose_1x3 = to_pose_matrix(xyz_1x3, rot_6d)
        pose_3 = to_pose_matrix(xyz_3, rot_6d)

        # Results should be equivalent (accounting for batch dimension)
        assert pose_1x3.shape == (1, 4, 4)
        assert pose_3.shape == (4, 4)
        np.testing.assert_allclose(pose_1x3[0], pose_3, rtol=1e-12)

    def test_large_batch_memory_layout(self):
        """Test that large batches maintain consistent memory layout."""
        large_batch_size = 1000
        xyz_batch = np.random.randn(large_batch_size, 3)
        rot_6d_batch = np.random.randn(large_batch_size, 6)

        # Should handle large batches without memory issues
        pose_matrices = to_pose_matrix(xyz_batch, rot_6d_batch)
        assert pose_matrices.shape == (large_batch_size, 4, 4)

        # Test random sample for correctness
        sample_indices = np.random.choice(large_batch_size, 10, replace=False)
        for idx in sample_indices:
            individual_pose = to_pose_matrix(xyz_batch[idx], rot_6d_batch[idx])
            np.testing.assert_allclose(pose_matrices[idx], individual_pose, rtol=1e-12)

    def test_nested_batch_operations(self):
        """Test operations that involve multiple levels of batching."""
        # Create sequence of poses (time series)
        time_steps = 10
        batch_size = 5

        # Shape: (batch_size, time_steps, 3) and (batch_size, time_steps, 6)
        xyz_sequences = np.random.randn(batch_size, time_steps, 3)
        rot_6d_sequences = np.random.randn(batch_size, time_steps, 6)

        # Test that we can process each sequence individually
        for b in range(batch_size):
            pose_sequence = to_pose_matrix(xyz_sequences[b], rot_6d_sequences[b])
            assert pose_sequence.shape == (time_steps, 4, 4)

            # Test relative poses within sequence
            reference_pose = pose_sequence[0]
            relative_poses = calculate_relative_pose(pose_sequence, reference_pose)
            assert relative_poses.shape == (time_steps, 4, 4)

            # First pose should be identity relative to itself
            np.testing.assert_allclose(relative_poses[0], np.eye(4), atol=1e-12)
