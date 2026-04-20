#!/usr/bin/env python3
"""
Gradio-based Robotics Data Explorer

This tool provides a gradio web interface for exploring preprocessed robotics data
with interactive trajectory visualization overlaid on camera images.

Usage:
    python vla_foundry/data/robotics/data_explorer_gradio.py \
        --dataset-path /path/to/processed/dataset/ \
        --max-samples 100 \
        --port 7860

Features:
    - Modern web-based interface using Gradio
    - Interactive sliders and dropdowns
    - Real-time trajectory overlay on camera images
    - 3D trajectory visualization
    - Multiple camera support
    - Gripper state visualization
    - Sample metadata display
"""

from typing import Any

import gradio as gr
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

from vla_foundry.data.robotics.gradio_dataloader import extract_trajectories


class CameraProjection:
    """Handle camera projection from 3D world coordinates to 2D image coordinates."""

    # Class variable to track extrinsics choices for consistency across samples
    _extrinsics_choices = {}

    def __init__(self, intrinsics: np.ndarray, extrinsics: np.ndarray):
        # Store original intrinsics and extrinsics
        self.intrinsics = intrinsics[0] if len(intrinsics.shape) == 3 else intrinsics  # Handle time-varying case
        self.extrinsics_original = (
            extrinsics[0] if len(extrinsics.shape) == 3 else extrinsics
        )  # Handle time-varying case
        self.extrinsics_inverted = np.linalg.inv(self.extrinsics_original)

        # Initially use original, but project_3d_to_2d will choose the best one
        self.extrinsics = self.extrinsics_original.copy()
        self.use_inverted = False

    def project_3d_to_2d(self, points_3d: np.ndarray, image_ratios: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """Project 3D points to 2D image coordinates, automatically choosing the best extrinsics."""
        # Test both original and inverted extrinsics
        # points_2d_orig = self._project_with_extrinsics(points_3d, self.extrinsics_original)
        points_2d_inv = self._project_with_extrinsics(points_3d, self.extrinsics_inverted)

        # Generate a camera identifier for consistency tracking
        camera_id = f"cam_{hash(str(self.intrinsics.tolist()) + str(self.extrinsics_original.tolist())) % 10000}"

        chosen_inverted = True

        # Store the choice for consistency
        CameraProjection._extrinsics_choices[camera_id] = chosen_inverted

        # Apply the chosen extrinsics and return the result
        self.extrinsics = self.extrinsics_inverted
        self.use_inverted = True
        result_points = points_2d_inv

        return result_points * np.array(image_ratios)[None, :]

    def project_3d_to_2d_scaled_intrinsics(
        self, points_3d: np.ndarray, image_ratios: tuple[float, float] = (1.0, 1.0)
    ) -> np.ndarray:
        """Project 3D points to 2D image coordinates using scaled intrinsics."""
        # Scale the intrinsics matrix to match the current image size
        # Camera intrinsics matrix format:
        # [fx  0  cx]
        # [0   fy cy]
        # [0   0   1]
        # where fx, fy are focal lengths and cx, cy is the principal point
        scaled_intrinsics = self.intrinsics.copy()

        # Apply scaling: fx and cx scale with width ratio, fy and cy scale with height ratio
        # For a camera intrinsics matrix:
        # [fx  0  cx]
        # [0   fy cy]
        # [0   0   1]
        # When scaling from original_size to current_size:
        # fx_new = fx_old * (current_width / original_width)
        # fy_new = fy_old * (current_height / original_height)
        # cx_new = cx_old * (current_width / original_width)
        # cy_new = cy_old * (current_height / original_height)
        scaled_intrinsics[0, 0] *= image_ratios[0]  # fx (focal length x)
        scaled_intrinsics[1, 1] *= image_ratios[1]  # fy (focal length y)
        scaled_intrinsics[0, 2] *= image_ratios[0]  # cx (principal point x)
        scaled_intrinsics[1, 2] *= image_ratios[1]  # cy (principal point y)

        # Use the same extrinsics choice as the main method
        self.extrinsics = self.extrinsics_inverted
        self.use_inverted = True

        # Project using scaled intrinsics
        points_3d_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
        points_cam = (self.extrinsics @ points_3d_homo.T).T[:, :3]
        points_2d_homo = (scaled_intrinsics @ points_cam.T).T

        # Convert from homogeneous to image coordinates
        z_coords = points_2d_homo[:, 2:3]
        z_coords = np.where(np.abs(z_coords) < 1e-8, 1e-8, z_coords)
        points_2d = points_2d_homo[:, :2] / z_coords

        return points_2d

    def _project_with_extrinsics(self, points_3d: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """Helper method to project using specific extrinsics matrix."""
        # Convert to homogeneous coordinates
        points_3d_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)

        # Transform to camera coordinates
        points_cam = (extrinsics @ points_3d_homo.T).T[:, :3]

        # Project to image coordinates
        points_2d_homo = (self.intrinsics @ points_cam.T).T

        # Convert from homogeneous to image coordinates
        # Avoid division by zero
        z_coords = points_2d_homo[:, 2:3]
        z_coords = np.where(np.abs(z_coords) < 1e-8, 1e-8, z_coords)
        points_2d = points_2d_homo[:, :2] / z_coords

        return points_2d

    def is_point_in_front(self, points_3d: np.ndarray) -> np.ndarray:
        """Check if 3D points are in front of the camera."""
        points_3d = np.asarray(points_3d)

        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)

        # Use the same extrinsics matrix that was selected in project_3d_to_2d
        # Convert to homogeneous coordinates
        points_3d_homo = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)

        # Transform to camera coordinates
        points_cam = (self.extrinsics @ points_3d_homo.T).T[:, :3]

        # Check if z > 0 (in front of camera)
        return points_cam[:, 2] > 0


class GradioDataExplorer:
    """Gradio-based data explorer."""

    def __init__(self, samples: list[dict[str, Any]], trajectory_length: int = 50):
        self.samples = samples
        self.trajectory_length = trajectory_length
        self.available_cameras = self._get_available_cameras()
        self.show_desired_trajectories = False
        self.show_action_trajectories = False
        self.use_reconstructed = False

    def _get_available_cameras(self) -> list[str]:
        """Get list of available camera names."""
        cameras = set()

        for sample in self.samples[:10]:
            if "images" in sample:
                for img_key in sample["images"]:
                    # Handle both formats:
                    # 1. Direct loading: scene_left_0_t-5, scene_left_0_t0, etc.
                    # 2. Dataloader: scene_left_0 (no timestep suffix)
                    if "_t" in img_key:
                        # Direct loading format: extract base camera name
                        camera_name = img_key.rsplit("_t", 1)[0]
                        cameras.add(camera_name)
                    else:
                        # Dataloader format: use the key directly as camera name
                        cameras.add(img_key)

        return sorted(list(cameras))

    def _get_available_camera_timesteps(self, camera_name: str) -> list[str]:
        """Get list of available timesteps for a specific camera."""
        timesteps = set()

        for sample in self.samples[:10]:
            if "images" in sample:
                for img_key in sample["images"]:
                    # Handle both formats:
                    # 1. Direct loading: scene_left_0_t-5, scene_left_0_t0, etc.
                    # 2. Dataloader: scene_left_0 (no timestep suffix)
                    if img_key.startswith(camera_name + "_t"):
                        # Direct loading format: extract offset from key format: camera_name_t{offset}
                        timestep_part = img_key[len(camera_name + "_t") :]
                        timesteps.add(f"t{timestep_part}")
                    elif img_key == camera_name:
                        # Dataloader format: single image per camera, use t0 as default
                        timesteps.add("t0")

        # Sort timesteps numerically by their offset values
        def sort_key(timestep):
            try:
                return int(timestep[1:])  # Remove 't' and convert to int
            except ValueError:
                return 0

        return sorted(list(timesteps), key=sort_key)

    def _get_camera_projection(
        self, sample: dict[str, Any], camera_name: str, calibration_timestep: int
    ) -> CameraProjection | None:
        """Get camera projection for specified camera at given timestep."""
        if not camera_name:
            return None

        intrinsics = None
        extrinsics = None

        # Look for intrinsics and extrinsics
        if "intrinsics" in sample and sample["intrinsics"]:
            if f"original_intrinsics.{camera_name}" in sample["intrinsics"]:
                intrinsics = sample["intrinsics"][f"original_intrinsics.{camera_name}"]
            elif f"intrinsics.{camera_name}" in sample["intrinsics"]:
                intrinsics = sample["intrinsics"][f"intrinsics.{camera_name}"]
            elif camera_name in sample["intrinsics"]:
                intrinsics = sample["intrinsics"][camera_name]

        if "extrinsics" in sample and sample["extrinsics"]:
            if f"extrinsics.{camera_name}" in sample["extrinsics"]:
                extrinsics = sample["extrinsics"][f"extrinsics.{camera_name}"]
            elif camera_name in sample["extrinsics"]:
                extrinsics = sample["extrinsics"][camera_name]

        if intrinsics is not None and extrinsics is not None:
            # Handle time-varying calibration
            if extrinsics.ndim == 3 and extrinsics.shape[0] > 1:
                # Use the provided calibration_timestep instead of anchor_idx
                if calibration_timestep < extrinsics.shape[0]:
                    extrinsics = extrinsics[calibration_timestep]
                else:
                    # Fallback to anchor_idx if calibration_timestep is out of bounds
                    metadata = sample.get("metadata", {})
                    anchor_idx = metadata.get("anchor_timestep", 0) - metadata.get("lowdim_start_timestep", 0)
                    if anchor_idx < extrinsics.shape[0]:
                        extrinsics = extrinsics[anchor_idx]

            if intrinsics.ndim == 3:
                # Use the provided calibration_timestep instead of anchor_idx
                if calibration_timestep < intrinsics.shape[0]:
                    intrinsics = intrinsics[calibration_timestep]
                else:
                    # Fallback to anchor_idx if calibration_timestep is out of bounds
                    metadata = sample.get("metadata", {})
                    anchor_idx = metadata.get("anchor_timestep", 0) - metadata.get("lowdim_start_timestep", 0)
                    if anchor_idx < intrinsics.shape[0]:
                        intrinsics = intrinsics[anchor_idx]

            return CameraProjection(intrinsics, extrinsics)
        else:
            print(
                f"🔍 Missing calibration data: intrinsics={intrinsics is not None}, extrinsics={extrinsics is not None}"
            )
            return None

    def _overlay_trajectory_on_image(
        self, image: Image.Image, sample: dict[str, Any], camera_name: str, image_timestep_info: str = "t0"
    ) -> Image.Image:
        """Overlay trajectory on camera image."""
        # Create a copy to draw on
        img_with_overlay = image.copy()

        # Try to get original image sizes from metadata
        if "original_image_sizes" in sample["metadata"]:
            if camera_name in sample["metadata"]["original_image_sizes"]:
                orig_size = sample["metadata"]["original_image_sizes"][camera_name]
                # Calculate ratio: current_size / original_size
                image_ratio = (image.width / orig_size[0], image.height / orig_size[1])
            else:
                image_ratio = (1.0, 1.0)
        else:
            print("🔍 No original_image_sizes found in metadata")
            # Try to infer from camera intrinsics if available
            if "intrinsics" in sample and f"original_intrinsics.{camera_name}" in sample["intrinsics"]:
                intrinsics = sample["intrinsics"][f"original_intrinsics.{camera_name}"]
            elif "intrinsics" in sample and f"intrinsics.{camera_name}" in sample["intrinsics"]:
                intrinsics = sample["intrinsics"][f"intrinsics.{camera_name}"]
            elif camera_name in sample["intrinsics"]:
                intrinsics = sample["intrinsics"][camera_name]
            else:
                intrinsics = None

            if intrinsics is not None and intrinsics.ndim >= 2:
                # Camera intrinsics matrix has focal length and principal point
                # The principal point (cx, cy) should be at the center of the original image
                if intrinsics.ndim == 3:
                    # Time-varying intrinsics, use the first one
                    intrinsics = intrinsics[0]

                # Principal point is at (cx, cy) in the intrinsics matrix
                cx = intrinsics[0, 2]  # Principal point x
                cy = intrinsics[1, 2]  # Principal point y

                # Estimate original image size from principal point
                # Principal point should be roughly at the center of the image
                estimated_orig_width = int(cx * 2)  # Principal point is roughly at center
                estimated_orig_height = int(cy * 2)

                image_ratio = (image.width / estimated_orig_width, image.height / estimated_orig_height)
            else:
                image_ratio = (1.0, 1.0)

        # Debug: Check if intrinsics are available and their properties
        if "intrinsics" in sample and f"original_intrinsics.{camera_name}" in sample["intrinsics"]:
            intrinsics = sample["intrinsics"][f"original_intrinsics.{camera_name}"]
        elif "intrinsics" in sample and f"intrinsics.{camera_name}" in sample["intrinsics"]:
            intrinsics = sample["intrinsics"][f"intrinsics.{camera_name}"]
        elif "intrinsics" in sample and camera_name in sample["intrinsics"]:
            intrinsics = sample["intrinsics"][camera_name]
        else:
            print(f"🔍 No intrinsics found for {camera_name}")

        draw = ImageDraw.Draw(img_with_overlay)

        # Determine which timestep this image corresponds to
        metadata = sample.get("metadata", {})
        anchor_index = metadata.get("anchor_timestep", 0) - metadata.get("lowdim_start_timestep", 0)

        # Parse the timestep offset from the format t{offset}
        image_offset = 0
        if image_timestep_info and image_timestep_info.startswith("t"):
            try:
                offset_str = image_timestep_info[1:]  # Remove 't' prefix
                image_offset = int(offset_str)
            except ValueError:
                pass

        # Get camera projection using the image timestep calibration
        # This ensures robot positions project correctly for the specific image timestep
        image_timestep_for_calibration = anchor_index + image_offset
        camera_projection = self._get_camera_projection(sample, camera_name, image_timestep_for_calibration)

        if not camera_projection:
            # Fallback: add a text overlay indicating no calibration data
            font = ImageFont.load_default()

            text = "Camera calibration data not available"
            text_color = "red"

            # Calculate text position (top-left corner with some padding)
            text_bbox = draw.textbbox((0, 0), text, font=font) if font else (0, 0, 200, 20)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Draw background rectangle for text
            padding = 5
            bg_coords = [10, 10, 10 + text_width + 2 * padding, 10 + text_height + 2 * padding]
            draw.rectangle(bg_coords, fill="black", outline="white")

            # Draw text
            draw.text((10 + padding, 10 + padding), text, fill=text_color, font=font)

            return img_with_overlay

        # Extract trajectories (gripper tip only), including desired if requested
        trajectories = extract_trajectories(
            sample,
            include_desired=self.show_desired_trajectories,
            include_action=self.show_action_trajectories,
            use_reconstructed=self.use_reconstructed,
        )

        # Filter trajectories to only include gripper tip position data (_gripper_xyz)
        xyz_trajectories = {}
        for traj_name, traj_data in trajectories.items():
            if (
                traj_data is not None
                and len(traj_data) > 0
                and ("_gripper_xyz" in traj_name)
                and traj_data.shape[1] == 3
            ):
                xyz_trajectories[traj_name] = traj_data

        # Helper function to convert colorscale to RGB
        def get_gradient_color(traj_name: str, time_value: float) -> str:
            """Get RGB color based on trajectory name and time value."""
            # Determine color based on trajectory name
            is_desired = "_desired" in traj_name
            is_action = "_action" in traj_name

            if "left" in traj_name.lower():
                if is_desired:
                    # Orange gradient for left arm desired
                    r = 255
                    g = int(165 * time_value)
                    b = 0
                    return f"rgb({r}, {g}, {b})"
                elif is_action:
                    # Green gradient for left arm action
                    r = 0
                    g = 255
                    b = int(100 * time_value)
                    return f"rgb({r}, {g}, {b})"
                else:
                    # Red gradient for left arm actual
                    r = 255
                    g = int(100 * time_value)
                    b = int(100 * time_value)
                    return f"rgb({r}, {g}, {b})"
            elif "right" in traj_name.lower():
                if is_desired:
                    # Purple gradient for right arm desired
                    r = int(128 + 127 * time_value)
                    g = 0
                    b = int(128 + 127 * time_value)
                    return f"rgb({r}, {g}, {b})"
                elif is_action:
                    # Yellow gradient for right arm action
                    r = 255
                    g = 255
                    b = int(100 * time_value)
                    return f"rgb({r}, {g}, {b})"
                else:
                    # Blue gradient for right arm actual
                    r = int(100 * time_value)
                    g = int(100 * time_value)
                    b = 255
                    return f"rgb({r}, {g}, {b})"
            else:
                # Default blue-white gradient
                white_intensity = int(255 * time_value)
                return f"rgb({white_intensity}, {white_intensity}, {255})"

        # Calculate the current robot index based on the image timestep offset
        # The image offset indicates which timestep the image corresponds to relative to anchor
        current_robot_index = anchor_index + image_offset

        for traj_name, traj_3d in xyz_trajectories.items():
            if traj_3d is None or len(traj_3d) == 0:
                continue

            # Save original trajectory info for gripper alignment
            subsampling_step = 1

            # Limit trajectory length
            if len(traj_3d) > self.trajectory_length:
                subsampling_step = len(traj_3d) // self.trajectory_length
                traj_3d = traj_3d[::subsampling_step]

            try:
                # Project 3D points to 2D using scaled intrinsics for better accuracy
                points_2d = camera_projection.project_3d_to_2d_scaled_intrinsics(traj_3d, image_ratio)
                is_in_front = camera_projection.is_point_in_front(traj_3d)

                # Ensure we have consistent shapes for indexing
                if points_2d.ndim == 1:
                    points_2d = points_2d.reshape(1, -1)

                # Verify shapes match before indexing
                if len(points_2d) != len(is_in_front):
                    print(
                        f"Error: Shape mismatch for {traj_name}: points_2d has {len(points_2d)} "
                        f"rows but is_in_front has {len(is_in_front)} elements"
                    )
                    continue

                # Filter points that are in front of camera
                valid_indices = is_in_front
                valid_points = points_2d[valid_indices]

                # Determine gripper states for this trajectory from arm gripper data
                if "_desired" in traj_name:
                    state_name = traj_name.replace("_gripper_xyz_desired", "_arm_gripper_desired")
                else:
                    state_name = traj_name.replace("_gripper_xyz", "_arm_gripper")
                gripper_states = trajectories.get(state_name)
                state_vals = None
                threshold = None
                if gripper_states is not None:
                    try:
                        arr = np.array(gripper_states)
                        # If multi-dimensional, average dimensions
                        state_vals = arr.mean(axis=1) if arr.ndim > 1 else arr

                        # Fixed threshold based on known gripper values:
                        # 0.0 = closed, 0.1 = open, so threshold should be 0.09
                        threshold = 0.09

                    except Exception as e:
                        print(f"   Error processing gripper states: {e}")
                        state_vals = None
                        threshold = None

                if len(valid_points) > 0:
                    # Create time values for gradient (inverted: start=1, end=0)
                    num_points = len(traj_3d)  # Use original trajectory length for time calculation
                    time_values = np.linspace(1, 0, num_points)

                    # Draw trajectory line with gradient
                    if len(valid_points) > 1:
                        for i in range(len(valid_points) - 1):
                            x1, y1 = int(valid_points[i][0]), int(valid_points[i][1])
                            x2, y2 = int(valid_points[i + 1][0]), int(valid_points[i + 1][1])

                            # Check bounds
                            if (
                                0 <= x1 < image.width
                                and 0 <= y1 < image.height
                                and 0 <= x2 < image.width
                                and 0 <= y2 < image.height
                            ):
                                # Calculate time value for this segment (use average of start and end points)
                                # Map valid point indices back to original trajectory indices
                                valid_point_indices = np.where(valid_indices)[0]
                                if i < len(valid_point_indices) and i + 1 < len(valid_point_indices):
                                    orig_idx1 = valid_point_indices[i]
                                    orig_idx2 = valid_point_indices[i + 1]
                                    time_val = (time_values[orig_idx1] + time_values[orig_idx2]) / 2
                                else:
                                    time_val = 0.5  # Fallback

                                segment_color = get_gradient_color(traj_name, time_val)
                                draw.line([(x1, y1), (x2, y2)], fill=segment_color, width=6)

                    # Draw markers along the trajectory
                    marker_interval = max(1, len(valid_points) // 8)  # Show ~8 markers along trajectory
                    for i in range(0, len(valid_points), marker_interval):
                        x, y = int(valid_points[i][0]), int(valid_points[i][1])
                        if 0 <= x < image.width and 0 <= y < image.height:
                            # Calculate time value for this marker
                            valid_point_indices = np.where(valid_indices)[0]
                            if i < len(valid_point_indices):
                                orig_idx = valid_point_indices[i]
                                time_val = time_values[orig_idx]
                            else:
                                time_val = 0.5  # Fallback

                            marker_color = get_gradient_color(traj_name, time_val)
                            # Draw small marker shape based on gripper state
                            if state_vals is not None and threshold is not None and i < len(valid_point_indices):
                                state_val = state_vals[orig_idx]
                                # Apply gripper state interpretation
                                # Values > 0.08 = open, values <= 0.08 = closed
                                is_open = state_val > threshold

                                if is_open:
                                    # Open gripper: circle marker
                                    draw.ellipse(
                                        [x - 3, y - 3, x + 3, y + 3], fill=marker_color, outline="black", width=1
                                    )
                                else:
                                    # Closed gripper: square marker
                                    draw.rectangle(
                                        [x - 3, y - 3, x + 3, y + 3], fill=marker_color, outline="black", width=1
                                    )
                            else:
                                # Default small circle
                                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=marker_color, outline="black", width=1)

                    # Initial and final markers removed
                    # Highlight the current robot position (corresponding to the image timestamp)
                    # Clamp current_robot_index to valid trajectory range
                    clamped_robot_index = max(0, min(current_robot_index, len(traj_3d) - 1))

                    if clamped_robot_index < len(valid_indices) and valid_indices[clamped_robot_index]:
                        # Find the index in the valid_points array
                        current_valid_idx = np.sum(valid_indices[: clamped_robot_index + 1]) - 1
                        if current_valid_idx >= 0 and current_valid_idx < len(valid_points):
                            x, y = int(valid_points[current_valid_idx][0]), int(valid_points[current_valid_idx][1])
                            if 0 <= x < image.width and 0 <= y < image.height:
                                # Get gradient color for current position
                                current_time_val = time_values[clamped_robot_index]
                                current_gradient_color = get_gradient_color(traj_name, current_time_val)

                                # Use orange outline for clamped positions (past/future beyond trajectory)
                                # Use yellow outline for exact timestamp match
                                outline_color = "orange" if current_robot_index != clamped_robot_index else "yellow"

                                # Draw medium marker for current robot position with shape based on gripper state
                                # Determine gripper state at this timestep
                                open_state = True  # Default assumption
                                if (
                                    state_vals is not None
                                    and threshold is not None
                                    and clamped_robot_index < len(state_vals)
                                ):
                                    current_state_val = state_vals[clamped_robot_index]
                                    open_state = current_state_val > threshold
                                # Outer marker for outline
                                if open_state:
                                    # Open gripper: circle outline
                                    draw.ellipse(
                                        [x - 5, y - 5, x + 5, y + 5], fill=outline_color, outline="black", width=1
                                    )
                                else:
                                    # Closed gripper: square outline
                                    draw.rectangle(
                                        [x - 5, y - 5, x + 5, y + 5], fill=outline_color, outline="black", width=1
                                    )
                                # Inner marker for fill
                                if open_state:
                                    draw.ellipse(
                                        [x - 3, y - 3, x + 3, y + 3],
                                        fill=current_gradient_color,
                                        outline="black",
                                        width=1,
                                    )
                                else:
                                    draw.rectangle(
                                        [x - 3, y - 3, x + 3, y + 3],
                                        fill=current_gradient_color,
                                        outline="black",
                                        width=1,
                                    )

            except Exception as e:
                print(f"Warning: Failed to project trajectory {traj_name}: {e}")
                continue

        return img_with_overlay

    def _create_3d_plot(self, sample: dict[str, Any]) -> go.Figure:
        """Create interactive 3D trajectory plot using Plotly."""
        fig = go.Figure()

        # Extract and filter gripper trajectories, including desired if requested
        trajectories = extract_trajectories(
            sample,
            include_desired=self.show_desired_trajectories,
            include_action=self.show_action_trajectories,
            use_reconstructed=self.use_reconstructed,
        )
        valid_trajectories = self._get_valid_gripper_trajectories(trajectories)

        if not valid_trajectories:
            return self._create_empty_3d_plot()

        # Calculate cubic axis ranges based on trajectory data
        axis_range = self._calculate_cubic_axis_range(valid_trajectories)

        # Add trajectory traces to the plot
        for traj_idx, (traj_name, traj_data) in enumerate(valid_trajectories):
            self._add_trajectory_trace(fig, traj_name, traj_data, traj_idx, sample)

        # Configure plot layout
        self._configure_3d_layout(fig, axis_range, len(valid_trajectories))

        return fig

    def _get_valid_gripper_trajectories(self, trajectories: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
        """Filter and return valid gripper trajectories."""
        valid_trajectories = []

        for traj_name, traj_data in trajectories.items():
            # Include both regular and desired gripper trajectories
            if (
                traj_data is not None
                and len(traj_data) > 0
                and ("_gripper_xyz" in traj_name)
                and traj_data.shape[1] == 3
            ):
                # Subsample if trajectory is too long
                if len(traj_data) > self.trajectory_length:
                    step = len(traj_data) // self.trajectory_length
                    traj_data = traj_data[::step]

                valid_trajectories.append((traj_name, traj_data))

        return valid_trajectories

    def _calculate_cubic_axis_range(self, valid_trajectories: list[tuple[str, np.ndarray]]) -> list[list[float]]:
        """Calculate cubic axis ranges based on trajectory data."""
        all_points = np.concatenate([traj for _, traj in valid_trajectories], axis=0)
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)

        # Calculate center and maximum range for cubic aspect ratio
        center = (max_coords + min_coords) / 2
        max_range = np.max(max_coords - min_coords)

        # Add padding and create cubic ranges
        padding = 0.1
        half_range = (max_range / 2) + padding if max_range > 0 else 0.5

        return [
            [center[0] - half_range, center[0] + half_range],  # X range
            [center[1] - half_range, center[1] + half_range],  # Y range
            [center[2] - half_range, center[2] + half_range],  # Z range
        ]

    def _get_colorscale_for_trajectory(self, traj_name: str) -> list:
        """Get appropriate colorscale based on trajectory name."""
        is_desired = "desired" in traj_name.lower()
        is_action = "action" in traj_name.lower()

        # Custom colorscales with inverted gradient (slightly lighter start to darker end)
        # Format: list of [position, color] pairs where position is 0.0 to 1.0
        if "left" in traj_name.lower():
            if is_desired:
                # Light orange to dark orange
                return [[0, "rgb(255, 165, 0)"], [1, "rgb(180, 80, 0)"]]
            if is_action:
                # Green gradient for left arm action
                return [[0, "rgb(0, 255, 0)"], [1, "rgb(0, 100, 0)"]]
            else:
                # Light red to dark red
                return [[0, "rgb(255, 100, 100)"], [1, "rgb(180, 0, 0)"]]
        elif "right" in traj_name.lower():
            if is_desired:
                # Light purple to dark purple
                return [[0, "rgb(200, 0, 200)"], [1, "rgb(80, 0, 80)"]]
            if is_action:
                # Yellow gradient for right arm action
                return [[0, "rgb(255, 255, 0)"], [1, "rgb(100, 100, 0)"]]
            else:
                # Light blue to dark blue
                return [[0, "rgb(100, 100, 255)"], [1, "rgb(0, 0, 180)"]]
        else:
            if is_desired:
                # Light yellow-green to dark yellow-green
                return [[0, "rgb(150, 255, 0)"], [1, "rgb(50, 100, 0)"]]
            elif is_action:
                # Light green to dark green
                return [[0, "rgb(0, 255, 100)"], [1, "rgb(0, 100, 0)"]]
            else:
                # Light green to dark green
                return [[0, "rgb(0, 255, 100)"], [1, "rgb(0, 100, 0)"]]

    def _add_trajectory_trace(
        self, fig: go.Figure, traj_name: str, traj_data: np.ndarray, traj_idx: int, sample: dict[str, Any]
    ) -> None:
        """Add a single trajectory trace to the figure."""
        # Clean the trajectory name for display
        is_desired = "_desired" in traj_name

        if is_desired:
            # For desired trajectories, remove both _gripper_xyz and _desired
            clean_name = traj_name.replace("_gripper_xyz_desired", " (Desired)")
        else:
            # For regular trajectories, just remove _gripper_xyz
            clean_name = traj_name.replace("_gripper_xyz", "")

        colorscale = self._get_colorscale_for_trajectory(traj_name)

        # Create time-based gradient
        time_values = np.linspace(0.1, 0.5, len(traj_data))

        # Add main trajectory trace
        fig.add_trace(
            go.Scatter3d(
                x=traj_data[:, 0],
                y=traj_data[:, 1],
                z=traj_data[:, 2],
                mode="lines+markers",
                line=dict(
                    color=time_values,
                    colorscale=colorscale,
                    width=6,
                ),
                marker=dict(size=4, color=time_values, colorscale=colorscale, showscale=False),
                name=clean_name,
                opacity=0.9,
                hovertemplate=(
                    f"{clean_name}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: "
                    f"%{{z:.3f}}<br>Time: %{{marker.color:.2f}}<extra></extra>"
                ),
            )
        )

        # Add current position marker (only for actual trajectories, not desired ones)
        if "_desired" not in traj_name:
            self._add_current_position_marker(fig, traj_data, time_values, colorscale, clean_name, sample)

    def _add_current_position_marker(
        self,
        fig: go.Figure,
        traj_data: np.ndarray,
        time_values: np.ndarray,
        colorscale: str,
        clean_name: str,
        sample: dict[str, Any],
    ) -> None:
        """Add current position marker to trajectory."""
        metadata = sample.get("metadata", {})
        anchor_idx = metadata.get("anchor_timestep", 0) - metadata.get("lowdim_start_timestep", 0)

        # Clamp index to trajectory range
        current_idx = max(0, min(anchor_idx, len(traj_data) - 1))
        current_point = traj_data[current_idx]
        current_time_value = time_values[current_idx]
        current_color = px.colors.sample_colorscale(colorscale, [current_time_value])[0]

        fig.add_trace(
            go.Scatter3d(
                x=[current_point[0]],
                y=[current_point[1]],
                z=[current_point[2]],
                mode="markers",
                marker=dict(size=10, color=current_color, symbol="circle", line=dict(color="black", width=2)),
                name=f"{clean_name} Current",
                showlegend=False,
                hovertemplate=(
                    f"{clean_name} Current (t={current_idx})<br>X: "
                    f"%{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>"
                ),
            )
        )

    def _configure_3d_layout(self, fig: go.Figure, axis_range: list[list[float]], num_trajectories: int) -> None:
        """Configure the 3D plot layout."""
        fig.update_layout(
            title="Robot Trajectory (3D) - Interactive with Time Gradient",
            scene=dict(
                xaxis_title="X (m)",
                xaxis=dict(range=axis_range[0], showgrid=True, gridcolor="lightgray"),
                yaxis_title="Y (m)",
                yaxis=dict(range=axis_range[1], showgrid=True, gridcolor="lightgray"),
                zaxis_title="Z (m)",
                zaxis=dict(range=axis_range[2], showgrid=True, gridcolor="lightgray"),
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=min(1000, 800 + num_trajectories * 60),
            height=600,
            margin=dict(l=0, r=20 + num_trajectories * 60, b=0, t=50),
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1),
            annotations=[
                dict(
                    text="🎨 Color gradients show time progression: bright=start → dark=end",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.05,
                    xanchor="center",
                    yanchor="top",
                    showarrow=False,
                    font=dict(size=12, color="gray"),
                )
            ],
        )

    def _create_empty_3d_plot(self) -> go.Figure:
        """Create an empty 3D plot when no trajectories are available."""
        fig = go.Figure()
        fig.update_layout(
            title="No trajectory data available",
            scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="cube"),
            width=800,
            height=600,
        )
        return fig

    def _format_metadata(self, sample: dict[str, Any]) -> str:
        """Format sample metadata for display."""
        lines = []

        # Basic sample info
        if "metadata" in sample and sample["metadata"]:
            metadata = sample["metadata"]
            lines.append("📋 **Sample Metadata:**")
            lines.append(f"- Sample ID: {metadata.get('sample_id', 'Unknown')}")
            lines.append(f"- Episode ID: {metadata.get('episode_id', 'Unknown')}")
            lines.append(f"- Anchor timestep: {metadata.get('anchor_timestep', 'Unknown')}")
            lines.append(
                f"- Initial timestep: {metadata.get('anchor_timestep', 0) - metadata.get('lowdim_start_timestep', 0)}"
            )
            lines.append(f"- Episode length: {metadata.get('original_episode_length', 'Unknown')}")

            # Add padding info if present
            if metadata.get("is_padded", False):
                past_pad = metadata.get("past_padding", 0)
                future_pad = metadata.get("future_padding", 0)
                lines.append(f"- Padding: {past_pad} past, {future_pad} future")

            lines.append("")

        # Camera info
        cameras = self._get_available_cameras()
        lines.append(f"📷 **Cameras:** {len(cameras)} available")
        for cam in cameras:
            lines.append(f"  - {cam}")
        lines.append("")

        # Add extrinsics choice information if available
        # if hasattr(CameraProjection, '_extrinsics_choices') and CameraProjection._extrinsics_choices:
        #     lines.append("🔧 **Camera Extrinsics Choices:**")
        #     for camera_id, use_inverted in CameraProjection._extrinsics_choices.items():
        #         choice_str = "inverted" if use_inverted else "original"
        #         lines.append(f"  - {camera_id}: {choice_str}")
        #     lines.append("")

        # Available data types
        if "lowdim" in sample and sample["lowdim"]:
            lines.append(f"📊 **Low-dim data:** {len(sample['lowdim'])} keys")

            # Group keys by type for better readability
            relative_keys = [k for k in sample["lowdim"] if "_relative" in k]
            robot_keys = [k for k in sample["lowdim"] if k.startswith("robot__") and k not in relative_keys]
            other_keys = [k for k in sample["lowdim"] if not k.startswith("robot__")]

            if robot_keys:
                lines.append(f"  - Robot state: {len(robot_keys)} keys")
            if other_keys:
                lines.append(f"  - Other: {len(other_keys)} keys")
            if relative_keys:
                lines.append(f"  - Relative coordinates: {len(relative_keys)} keys")
                lines.append("    🔄 Using reconstructed absolute coordinates for visualization")
            if "language_instruction" in sample["lowdim"]:
                # Instruction is a list of strings when batched with a dataloader, but a string if read from a file
                instruction = (
                    sample["lowdim"]["language_instruction"]
                    if isinstance(sample["lowdim"]["language_instruction"], str)
                    else sample["lowdim"]["language_instruction"][0]
                )
                lines.append(f"  - Language instruction: {instruction}")

        return "\n".join(lines)

    def update_display(
        self, sample_idx: int, camera_name: str, camera_timestep: str, show_3d_plot: bool
    ) -> tuple[Image.Image, go.Figure | None, str]:
        """Update the display based on current selections."""
        if sample_idx >= len(self.samples):
            return None, None, "Invalid sample index"

        sample = self.samples[sample_idx]

        # Get camera image for the specific timestep
        image = None
        image_timestep_info = camera_timestep
        if camera_name and camera_timestep and "images" in sample:
            # Handle both formats:
            # 1. Direct loading: camera_name_t{offset} (e.g., "scene_left_0_t-5")
            # 2. Dataloader: camera_name (no timestep suffix)
            img_key = f"{camera_name}_{camera_timestep}"

            if img_key in sample["images"]:
                # Direct loading format: found exact match
                image = sample["images"][img_key]
            elif camera_name in sample["images"]:
                # Dataloader format: single image per camera
                image = sample["images"][camera_name]
                image_timestep_info = "t0"  # Use t0 as default for dataloader
            else:
                # Fallback: find any image for this camera (direct loading format)
                for img_key in sample["images"]:
                    if img_key.startswith(camera_name + "_t"):
                        image = sample["images"][img_key]
                        timestep_part = img_key[len(camera_name + "_t") :]
                        image_timestep_info = f"t{timestep_part}"
                        break

        if image is None:
            # Create placeholder image
            image = Image.new("RGB", (224, 224), color="lightgray")
            draw = ImageDraw.Draw(image)
            draw.text((112, 112), f"No image available\nfor camera: {camera_name}", fill="black", anchor="mm")
        else:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)

            # Overlay trajectory
            image = self._overlay_trajectory_on_image(image, sample, camera_name, image_timestep_info)

            # Resize image to match display size (e.g., 800x800)
            target_size = (600, 600)
            if image.size != target_size:
                image = image.resize(target_size, Image.LANCZOS)

        # Create 3D plot if requested
        plot_3d = None
        if show_3d_plot:
            plot_3d = self._create_3d_plot(sample)

        # Format metadata
        metadata_str = self._format_metadata(sample)

        return image, plot_3d, metadata_str

    def create_interface(self) -> gr.Interface:
        """Create the Gradio interface."""

        def update_fn(
            sample_idx, camera_name, camera_timestep, show_3d_plot, show_desired, show_action, use_reconstructed
        ):
            self.show_desired_trajectories = show_desired
            self.show_action_trajectories = show_action
            self.use_reconstructed = use_reconstructed
            return self.update_display(sample_idx, camera_name, camera_timestep, show_3d_plot)

        def update_camera_timesteps(camera_name):
            if camera_name:
                available_timesteps = self._get_available_camera_timesteps(camera_name)
                return gr.Dropdown(
                    choices=available_timesteps, value=available_timesteps[0] if available_timesteps else None
                )
            return gr.Dropdown(choices=[], value=None)

        # Create interface components
        with gr.Blocks(title="Robotics Data Explorer", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Robotics Data Explorer")
            gr.Markdown("Explore your robotics dataset with interactive trajectory visualization")

            with gr.Row():
                with gr.Column(scale=2):
                    # Controls
                    sample_slider = gr.Slider(
                        minimum=0,
                        maximum=len(self.samples) - 1,
                        step=1,
                        value=0,
                        label=f"Sample Index (0 to {len(self.samples) - 1})",
                    )

                    camera_dropdown = gr.Dropdown(
                        choices=self.available_cameras,
                        value=self.available_cameras[0] if self.available_cameras else None,
                        label="Camera",
                    )

                    # Get initial timesteps for the first camera
                    initial_timesteps = (
                        self._get_available_camera_timesteps(self.available_cameras[0])
                        if self.available_cameras
                        else []
                    )
                    camera_timestep_dropdown = gr.Dropdown(
                        choices=initial_timesteps,
                        value=initial_timesteps[-1] if initial_timesteps else None,
                        label="Camera Timestep",
                    )

                    show_3d_checkbox = gr.Checkbox(value=True, label="Show 3D Plot")
                    show_desired_checkbox = gr.Checkbox(value=False, label="Show Desired Trajectories")
                    show_action_checkbox = gr.Checkbox(value=False, label="Show Action Trajectories")
                    show_reconstructed_checkbox = gr.Checkbox(value=False, label="Use Reconstructed Coordinates")

                with gr.Column(scale=1):
                    # Metadata display
                    metadata_display = gr.Markdown(value="Select a sample to view metadata", label="Sample Metadata")

            with gr.Row():
                with gr.Column():
                    # Camera image with trajectory overlay
                    camera_image = gr.Image(type="pil", width=600, height=600)

                with gr.Column():
                    # 3D trajectory plot
                    trajectory_plot = gr.Plot(label="3D Trajectory View")

            # Add legend
            gr.Markdown(
                """
            ### 🎨 Legend
            - **Red trajectory**: Left arm actual position
            - **Blue trajectory**: Right arm actual position
            - **Orange trajectory**: Left arm desired position (when "Show Desired Trajectories" is enabled)
            - **Purple trajectory**: Right arm desired position (when "Show Desired Trajectories" is enabled)
            - **Yellow trajectory**: Actions (when "Show Actions" is enabled)
            - **Green trajectory**: Base (if available)
            - **🔵 Circle markers**: Open gripper state (value > 0.09)
            - **🔲 Square markers**: Closed gripper state (value ≤ 0.09)
            - **🟡 Yellow outline**: Current robot position (corresponding to image timestamp)
            - **🟢 Green trajectory**: Left arm action position (when "Show Action Trajectories" is enabled)
            - **🟡 Yellow trajectory**: Right arm action position (when "Show Action Trajectories" is enabled)
            ### 💡 Tips
            - The 3D plot shows interactive trajectory visualization with time-based color gradients
            - Use the camera timestep dropdown to see different temporal views of the same scene
            - Gripper states: 0.1 = open, 0.0 = closed (threshold = 0.09)
            - Enable "Show Desired Trajectories" to visualize both actual and desired robot positions
            """
            )

            # Connect all inputs to the update function
            inputs = [
                sample_slider,
                camera_dropdown,
                camera_timestep_dropdown,
                show_3d_checkbox,
                show_desired_checkbox,
                show_action_checkbox,
                show_reconstructed_checkbox,
            ]
            outputs = [camera_image, trajectory_plot, metadata_display]

            # Update camera timesteps when camera changes
            camera_dropdown.change(
                fn=update_camera_timesteps,
                inputs=[camera_dropdown],
                outputs=[camera_timestep_dropdown],
                show_progress=False,
            )

            # Update on any input change
            for input_component in inputs:
                input_component.change(fn=update_fn, inputs=inputs, outputs=outputs, show_progress=False)

            # Initialize display
            interface.load(fn=update_fn, inputs=inputs, outputs=outputs, show_progress=False)

        return interface
