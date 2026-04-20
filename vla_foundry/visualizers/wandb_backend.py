"""
WandB Backend Implementation

This file contains the implementation for the WandbBackend class, which provides
logging functionality to Weights & Biases (wandb).
"""

import os
from typing import Any

import numpy as np
import plotly.graph_objects as go
import wandb


class WandbBackend:
    name = "wandb"

    def __init__(self) -> None:
        """
        Initialize the WandbBackend instance.
        """
        self._initialized = False
        self._run = None

    def init(self, run_name: str, **kwargs) -> None:
        """
        Initialize the WandB visualizer.

        Parameters:
        - run_name: Name of the run.
        - **kwargs: Additional arguments passed to wandb.init().
        """
        if not self._initialized:
            project = os.environ.get("WANDB_PROJECT")
            if not project:
                raise ValueError(
                    "WANDB_PROJECT environment variable must be set when using the wandb backend. "
                    "Example: export WANDB_PROJECT=my_project"
                )
            self._run = wandb.init(project=project, name=run_name)
            self._initialized = True

    def flush(self) -> None:
        """
        Flush any pending logs to the WandB backend.
        """
        if self._run:
            self._run.log({})  # WandB automatically flushes logs; this is a no-op.

    def shutdown(self) -> None:
        """
        Shutdown the WandB backend and clean up resources.
        """
        if self._run:
            self._run.finish()
            self._run = None
            self._initialized = False

    def log_images(self, path: str, images: Any, **kwargs) -> None:
        """
        Log images to the WandB backend. Supports both single images and dictionaries of images.

        Args:
            path: Base path in the visualization hierarchy.
            images: Either a single NumPy array representing an image or a dictionary of images.
        """
        if isinstance(images, np.ndarray):
            wandb.log({path: wandb.Image(images)})
        elif isinstance(images, dict):
            for sub_path, image in images.items():
                wandb.log({f"{path}/{sub_path}": wandb.Image(image)})
        else:
            raise TypeError("Unsupported type for 'images'. Must be a NumPy array or a dictionary of NumPy arrays.")

    def log_scalar(self, path: str, value: float, **kwargs) -> None:
        """
        Log a scalar value to the WandB backend.

        Parameters:
        - path: The hierarchical path for the scalar.
        - value: The scalar value.
        """
        wandb.log({path: value})

    def log_points3d(self, path: str, points: np.ndarray, **kwargs) -> None:
        """
        Log 3D points to the WandB backend using Plotly.

        Parameters:
        - path: The hierarchical path for the 3D points.
        - points: The 3D points as a NumPy array of shape (N, 3).
        """
        fig = go.Figure(
            data=[go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", marker=dict(size=5))]
        )
        # Set consistent axis ranges based on data bounds
        padding = 0.1 * max(points.max() - points.min(), 1.0)
        fig.update_layout(
            title=f"3D Points: {path}",
            scene=dict(
                aspectmode="cube",
                xaxis=dict(title="X", range=[points[:, 0].min() - padding, points[:, 0].max() + padding]),
                yaxis=dict(title="Y", range=[points[:, 1].min() - padding, points[:, 1].max() + padding]),
                zaxis=dict(title="Z", range=[points[:, 2].min() - padding, points[:, 2].max() + padding]),
            ),
        )
        wandb.log({path: fig})

    def log_line_strips3d(self, path: str, line_strips: np.ndarray | dict[str, np.ndarray], **kwargs) -> None:
        """
        Log 3D line strips to the WandB backend using Plotly.

        Supports:
        - Single line strip as np.ndarray of shape (N, 3)
        - Multiple named line strips as dict[str, np.ndarray]

        Parameters:
        - path: The hierarchical path for the line strips.
        - line_strips: The 3D line strips.
        """
        lines: dict[str, np.ndarray]
        if isinstance(line_strips, np.ndarray):
            lines = {"trajectory": line_strips}
        elif isinstance(line_strips, dict):
            lines = {str(name): np.asarray(points) for name, points in line_strips.items()}
        else:
            raise TypeError("line_strips must be np.ndarray or dict[str, np.ndarray]")

        colors = kwargs.get("colors", {})
        default_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
        fig = go.Figure()
        all_points: list[np.ndarray] = []

        for idx, (name, points) in enumerate(lines.items()):
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Line strip '{name}' must have shape (N, 3), got {points.shape}")
            color = colors.get(name, default_colors[idx % len(default_colors)])
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="lines+markers",
                    line=dict(width=5, color=color),
                    marker=dict(size=2, color=color),
                    name=name,
                    showlegend=True,
                )
            )
            all_points.append(points)

        if not all_points:
            raise ValueError("No line strips to log")

        stacked_points = np.concatenate(all_points, axis=0)
        # Set consistent axis ranges based on data bounds
        padding = 0.1 * max(stacked_points.max() - stacked_points.min(), 1.0)
        fig.update_layout(
            title=f"3D Line Strip: {path}",
            scene=dict(
                aspectmode="cube",
                xaxis=dict(
                    title="X",
                    range=[stacked_points[:, 0].min() - padding, stacked_points[:, 0].max() + padding],
                ),
                yaxis=dict(
                    title="Y",
                    range=[stacked_points[:, 1].min() - padding, stacked_points[:, 1].max() + padding],
                ),
                zaxis=dict(
                    title="Z",
                    range=[stacked_points[:, 2].min() - padding, stacked_points[:, 2].max() + padding],
                ),
            ),
        )
        wandb.log({path: fig})

    def log_pose(
        self, path: str, translation: np.ndarray, rotation: np.ndarray, axis_length: float = 1.0, **kwargs
    ) -> None:
        """
        Log a generic pose to the WandB backend using Plotly 3D visualization.

        Args:
            path: Path in the visualization hierarchy.
            translation: Translation vector of shape (3,).
            rotation: Quaternion [x, y, z, w] of shape (4,).
            axis_length: Length of the axes for visualization, by default 1.0.
        """
        # Ensure translation is a 1D numpy array of shape (3,)
        translation = np.asarray(translation)
        if translation.shape != (3,):
            raise ValueError(f"Translation must be shape (3,), got {translation.shape} and type {type(translation)}")
        rotation = np.asarray(rotation)

        # Convert quaternion to rotation matrix for axis visualization
        # Note: rotation is [x, y, z, w] format from visualizer facade
        from scipy.spatial.transform import Rotation

        rot_matrix = Rotation.from_quat(rotation).as_matrix()

        # Create coordinate frame axes
        origin = translation
        x_axis = origin + rot_matrix[:, 0] * axis_length  # First column = X axis
        y_axis = origin + rot_matrix[:, 1] * axis_length  # Second column = Y axis
        z_axis = origin + rot_matrix[:, 2] * axis_length  # Third column = Z axis

        # Create Plotly figure with coordinate frame axes (like Rerun)
        fig = go.Figure()

        # Add X axis (red) - from origin to X direction
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], x_axis[0]],
                y=[origin[1], x_axis[1]],
                z=[origin[2], x_axis[2]],
                mode="lines",
                line=dict(color="red", width=8),
                name="X",
                showlegend=False,
            )
        )

        # Add Y axis (green) - from origin to Y direction
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], y_axis[0]],
                y=[origin[1], y_axis[1]],
                z=[origin[2], y_axis[2]],
                mode="lines",
                line=dict(color="green", width=8),
                name="Y",
                showlegend=False,
            )
        )

        # Add Z axis (blue) - from origin to Z direction
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], z_axis[0]],
                y=[origin[1], z_axis[1]],
                z=[origin[2], z_axis[2]],
                mode="lines",
                line=dict(color="blue", width=8),
                name="Z",
                showlegend=False,
            )
        )

        # Add small sphere at origin
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0]],
                y=[origin[1]],
                z=[origin[2]],
                mode="markers",
                marker=dict(size=6, color="black"),
                name="Origin",
                showlegend=False,
            )
        )

        fig.update_layout(
            title=f"Pose: {path}",
            scene=dict(
                aspectmode="cube",
                xaxis=dict(title="X", range=[origin[0] - axis_length * 1.5, origin[0] + axis_length * 1.5]),
                yaxis=dict(title="Y", range=[origin[1] - axis_length * 1.5, origin[1] + axis_length * 1.5]),
                zaxis=dict(title="Z", range=[origin[2] - axis_length * 1.5, origin[2] + axis_length * 1.5]),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=30),
        )
        wandb.log({path: fig})

        # Also log individual scalars for table visibility
        wandb.log(
            {
                f"{path}/translation_x": float(translation[0]),
                f"{path}/translation_y": float(translation[1]),
                f"{path}/translation_z": float(translation[2]),
                f"{path}/quaternion_x": float(rotation[0]),
                f"{path}/quaternion_y": float(rotation[1]),
                f"{path}/quaternion_z": float(rotation[2]),
                f"{path}/quaternion_w": float(rotation[3]),
            }
        )

    def log_text(self, path: str, text: str, **kwargs) -> None:
        """
        Log a text value to the WandB backend.

        Args:
            path: Path in the visualization hierarchy.
            text: Text value to log.
        """
        wandb.log({path: text})
