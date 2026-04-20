"""
Tutorial: How to Use the Visualizer Module

This script demonstrates how to use the `visualizer` module for logging images, scalars, 3D points, trajectories,
and rigid transforms. It also shows how to use the methods for logging robot arm poses and model action predictions.

Make sure the `VISUALIZER` environment variable is set to `rerun` or `disabled` before running this script.
"""

import random
import time

import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw
from robot_gym.multiarm_spaces import PosesAndGrippers  # Import PosesAndGrippers from robot_gym.multiarm_spaces

import vla_foundry.visualizers.visualizer as vz

# Initialize the visualizer
vz.init(run_name="tutorial_logging", add_rank_to_run=True)

print("\n--- Sparse logging demo ---")
print("n=5 logs on call 1, then 5, 10, 15... per key")

for step in range(1, 16):
    vz.log_scalar("every_n/n5", step, n=5)
    vz.log_images("every_n/img_n5", np.random.rand(64, 64, 3), n=5)

print("\n--- Runtime enable/disable demo ---")
vz.disable()
vz.log_scalar("toggle/only_when_enabled", -1)  # should NOT appear
vz.enable()
vz.log_scalar("toggle/only_when_enabled", 1)  # should appear


# Loop to log dynamic data
for step in range(10):
    print(f"Step {step + 1}/10")

    # 1a. Log a random image
    print("Logging a random image...")
    image = np.random.randint(0, 256, (255, 255, 3), dtype=np.uint8)  # Random image
    vz.log_images("single_image", image)  # Updated to use log_images for single image

    # 1b. Log a dict of images
    print("Logging a dictionary of images...")
    image_dict = {}
    for ii in range(5):
        image = np.random.randint(0, 256, (255, 255, 3), dtype=np.uint8)  # Random image
        image_dict[f"image_{ii}"] = image
    vz.log_images("image_dict", image_dict)

    # 2. Log a scalar value that changes over time
    print("Logging a scalar value...")
    vz.log_scalar("metrics/loss", random.uniform(0.1, 1.0))

    # 3. Log 3D points with random positions
    print("Logging random 3D points...")
    points = np.random.rand(5, 3) * 10  # Random points in a 10x10x10 cube
    vz.log_points3d("points", points)

    # 4. Log a trajectory that changes shape
    print("Logging a dynamic trajectory...")
    trajectory = np.cumsum(np.random.randn(10, 3), axis=0)  # Random walk in 3D
    vz.log_trajectory("trajectory", trajectory)

    # 5. Log a rigid transform with random translation and rotation
    print("Logging a random rigid transform...")
    pose = RigidTransform(
        RollPitchYaw(*np.random.uniform(0, np.pi, 3)),  # Random rotation
        np.random.uniform(-5, 5, 3),  # Random translation
    )
    vz.log_rigid_transform("robot/pose", pose, axis_length=0.5)

    # 5b. Log generic poses with different formats
    print("Logging generic poses with different formats...")

    # Example 1: Using quaternion [x, y, z, w]
    translation = np.random.uniform(-2, 2, 3)
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    vz.log_pose("generic_poses/quaternion", translation, quaternion)

    # Example 2: Using rotation matrix
    rotation_matrix = np.eye(3)  # Identity rotation
    translation = np.random.uniform(-2, 2, 3)
    vz.log_pose("generic_poses/rotation_matrix", translation, rotation_matrix)

    # Example 3: Using 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = np.random.uniform(-2, 2, 3)  # Random translation
    vz.log_pose("generic_poses/transform_matrix", np.zeros(3), transform_matrix)  # translation ignored

    # 6. Log robot poses and grippers with random configurations
    print("Logging random robot poses and grippers...")
    poses_and_grippers = PosesAndGrippers(
        poses={
            "arm_1": RigidTransform(RollPitchYaw(0, 0, 0), np.random.uniform(-1, 1, 3)),
        },
        grippers={"gripper_1": random.uniform(0, 1)},
    )
    vz.log_robot_gym_poses_and_grippers("robot_gym/poses_and_grippers", poses_and_grippers)

    # 7. Log model action predictions with random data
    print("Logging random model action predictions...")
    action_predictions = [
        PosesAndGrippers(
            poses={
                "arm_1": RigidTransform(RollPitchYaw(0, 0, 0), np.random.uniform(-1, 1, 3)),
            },
            grippers={"gripper_1": random.uniform(0, 1)},
        )
        for _ in range(2)
    ]
    vz.log_robot_gym_action_predictions("robot_gym/action_predictions", action_predictions)

    # Simulate time delay between steps
    time.sleep(1)

print("Tutorial complete!")
