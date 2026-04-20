"""
CV utilities.

This module provides helper functions for working with computer vision data,
including rescaling of camera intrinsics to match image preprocessing steps.
"""

from typing import Any

import matplotlib.cm as cm
import numpy as np

from vla_foundry.data.preprocessing.image_utils import ImageResizingMethod


def intrinsics_3x3_to_4(K: np.ndarray) -> np.ndarray:
    """
    Convert camera intrinsics from shape (3,3) or (N,3,3)
    to (fx, fy, cx, cy) with shape (4,) or (N,4).
    """
    K = np.asarray(K)

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"Expected shape (..., 3, 3), got {K.shape}")

    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    return np.stack([fx, fy, cx, cy], axis=-1).astype(float)


def intrinsics_4_to_3x3(intrinsics: np.ndarray) -> np.ndarray:
    """
    Convert camera intrinsics from (fx, fy, cx, cy) with shape (4,) or (N,4)
    to intrinsic matrix/matrices with shape (3,3) or (N,3,3).
    """
    intr = np.asarray(intrinsics)

    if intr.ndim not in (1, 2):
        raise ValueError(f"intrinsics must have shape (4,) or (N,4), got {intr.shape}")
    if intr.shape[-1] != 4:
        raise ValueError(f"Last dimension must be 4, got {intr.shape}")

    fx, fy, cx, cy = intr[..., 0], intr[..., 1], intr[..., 2], intr[..., 3]

    # Create output array
    out_shape = intr.shape[:-1] + (3, 3)
    K = np.zeros(out_shape, dtype=float)

    K[..., 0, 0] = fx
    K[..., 1, 1] = fy
    K[..., 0, 2] = cx
    K[..., 1, 2] = cy
    K[..., 2, 2] = 1.0

    return K


def scale_intrinsics_4_for_resize_and_crop(
    original_intrinsics: np.ndarray,
    original_image_size: tuple[int, int],
    processed_image_size: tuple[int, int],
    resize_method: ImageResizingMethod = ImageResizingMethod.CENTER_CROP,
) -> np.ndarray:
    """Scales camera intrinsics to account for resizing and cropping/padding.

    Args:
        original_intrinsics: Camera intrinsics as (fx, fy, cx, cy) or (N, 4).
        original_image_size: Original image size as (width, height).
        processed_image_size: Processed image size as (width, height).
        resize_method: Method used to resize the image.

    Returns
    -------
    np.ndarray
        Scaled intrinsics with same shape as input: (4,) or (N, 4).
    """
    original_intrinsics = np.asarray(original_intrinsics)
    if original_intrinsics.ndim not in (1, 2):
        raise ValueError(f"original_intrinsics must have ndim 1 or 2, got {original_intrinsics.ndim}")
    if original_intrinsics.shape[-1] != 4:
        raise ValueError(f"original_intrinsics last dim must be 4, got {original_intrinsics.shape}")

    W0, H0 = original_image_size
    W, H = processed_image_size

    fx, fy, cx, cy = (
        original_intrinsics[..., 0],
        original_intrinsics[..., 1],
        original_intrinsics[..., 2],
        original_intrinsics[..., 3],
    )

    if resize_method == ImageResizingMethod.CENTER_CROP:
        # Scale to cover target
        scale = max(W / W0, H / H0)

        new_W = W0 * scale
        new_H = H0 * scale

        # Center crop offsets
        cx_offset = (new_W - W) / 2.0
        cy_offset = (new_H - H) / 2.0

        fx1 = fx * scale
        fy1 = fy * scale
        cx1 = cx * scale - cx_offset
        cy1 = cy * scale - cy_offset

    elif resize_method == ImageResizingMethod.RESIZE_NO_CROP:
        # Independent scaling (may distort)
        sx = W / W0
        sy = H / H0

        fx1 = fx * sx
        fy1 = fy * sy
        cx1 = cx * sx
        cy1 = cy * sy

    elif resize_method == ImageResizingMethod.RESIZE_FIT:
        # Scale to fit inside target
        scale = min(W / W0, H / H0)

        new_W = W0 * scale
        new_H = H0 * scale

        # Padding offsets (image pasted centered)
        pad_x = (W - new_W) / 2.0
        pad_y = (H - new_H) / 2.0

        fx1 = fx * scale
        fy1 = fy * scale
        cx1 = cx * scale + pad_x
        cy1 = cy * scale + pad_y

    else:
        raise ValueError(f"Unrecognized image resizing method: {resize_method}")

    return np.stack([fx1, fy1, cx1, cy1], axis=-1)


def scale_intrinsics_3x3_for_resize_and_crop(
    original_intrinsics: np.ndarray,
    original_image_size: tuple[int, int],
    processed_image_size: tuple[int, int],
    resize_method: ImageResizingMethod = ImageResizingMethod.CENTER_CROP,
) -> np.ndarray:
    """Scales camera intrinsics to account for resizing and cropping.

    Parameters
    ----------
    original_intrinsics : np.ndarray
        Camera intrinsics containing (fx, fy, cx, cy). Accepts shape (3, 3) or (N, 3, 3).
    original_image_size : tuple[int, int]
        Original image size as (width, height).
    processed_image_size : tuple[int, int]
        Processed image size as (width, height).

    Returns
    -------
    np.ndarray
        Scaled intrinsics with same shape as input: (3, 3) or (N, 3, 3).
    """
    return intrinsics_4_to_3x3(
        scale_intrinsics_4_for_resize_and_crop(
            intrinsics_3x3_to_4(original_intrinsics), original_image_size, processed_image_size, resize_method
        )
    )


def transform_points_to_camera_frame(camera_T_base: np.ndarray, base_t_pts: np.ndarray) -> np.ndarray:
    """Transform 3D points from base frame to camera frame.

    Args:
        camera_T_base: Homogeneous transformation matrix from base frame to camera frame of shape (4, 4).
        base_t_pts: Points in base frame of shape (N, 3).

    Returns:
        Points in camera frame of shape (N, 3).
    """

    base_t_pts_homogeneous = np.concatenate(
        [base_t_pts, np.ones((base_t_pts.shape[0], 1), dtype=base_t_pts.dtype)], axis=1
    )
    camera_t_pts_homogeneous = (camera_T_base @ base_t_pts_homogeneous.T).T
    camera_t_pts = camera_t_pts_homogeneous[:, :-1]
    return camera_t_pts


def draw_circle(img: np.ndarray, center: tuple[int, int], radius: int, color: int | tuple[int, int, int]):
    """Draw a circle onto an image.

    Args:
        img: Numpy array for RGB or depth images respectively, of shape (H, W, 3) or (H, W).
        center: Coordinates of the circle's center as (x, y).
        radius: Radius of the circle.
        color: Color value for the circle.
    """
    out = np.array(img, copy=True)
    x0, y0 = center
    h, w = out.shape[:2]

    y, x = np.ogrid[:h, :w]
    mask = (x - x0) ** 2 + (y - y0) ** 2 <= radius**2

    out[mask] = color
    return out


def draw_projected_trajectory_if_rgb(
    image: np.ndarray,
    intrinsics: np.ndarray,
    trace_pts_list: list[np.ndarray],
):
    """Draw a 3D trajectory as a set of circles projected onto the image, if the image is RGB.

    Args:
        image: Numpy array for RGB or depth images respectively, of shape (H, W, 3) or (H, W).
        intrinsics: Camera intrinsics containing (fx, fy, cx, cy). This function expects to unpack them as
            `fx, fy, cx, cy = intrinsics`.
        trace_pts_list: List of 3D trajectories (N, 3) in the camera frame that will be
            projected onto the 2D image. Each trajectory is plotted independently from each
            other, and can come from differents sources (e.g. right arm and left arm).
    """
    # Only handle RGB for now, and not depth.
    if image.ndim != 3 or image.shape[2] < 3:
        return image
    img_out = image

    # Get camera intrinsics.
    fx, fy, cx, cy = intrinsics

    for trace_pts in trace_pts_list:
        X, Y, Z = trace_pts[:, 0], trace_pts[:, 1], trace_pts[:, 2]
        valid = Z > 0

        Xv, Yv, Zv = X[valid], Y[valid], Z[valid]

        u = fx * (Xv / Zv) + cx
        v = fy * (Yv / Zv) + cy

        # Radius scaling so that a 3D trace_pt farther away appears smaller when projected.
        R_m = 0.01  # Radius of the "real" sphere in meters.
        r_min, r_max = 3, 7  # Min and max radii in the 2D image.
        f_eff = 0.5 * (fx + fy)
        r_px = f_eff * (R_m / Zv)
        r_px = np.clip(r_px, r_min, r_max)

        circle_coords = np.stack([u, v], axis=1)
        circle_coords = np.round(circle_coords).astype(np.int32)

        cmap = cm.get_cmap("viridis")
        num_trace_pts = len(circle_coords)

        # Sort all trace points by depth so that closer points are drawn last.
        order = np.argsort(Zv)[::-1]  # Indices for decreasing Zv.
        circle_coords_sorted = circle_coords[order]
        r_px_sorted = r_px[order]

        for circle_coord, r_p, orig_idx in zip(circle_coords_sorted, r_px_sorted, order, strict=True):
            trace_idx = orig_idx / max(num_trace_pts - 1, 1)
            rgba = cmap(trace_idx)
            rgb = tuple((np.array(rgba[:3]) * 255).astype(np.uint8))
            img_out = draw_circle(img_out, circle_coord, r_p, rgb)

    return img_out


def create_images_with_projected_trace(
    images: Any,
    intrinsics: np.ndarray,
    trace_pts_list_or_dict: list[np.ndarray] | dict[str, list[np.ndarray]],
    **kwargs,
) -> None:
    """Create images with projected traces.

    Args:
        images: Either a single NumPy array representing an image or a dictionary of images.
        intrinsics: Either a single NumPy array of amera intrinsics containing (fx, fy, cx, cy) = intrinsics,
            or a dictionary of intrinsics.
        trace_pts_list_or_dict: List (or dictionary of lists) of 3D trajectories (N, 3) in the camera frame
            that will be projected onto the 2D image. Each trajectory is plotted independently from each
            other, and can come from differents sources (e.g. right arm and left arm).
    """

    # Project 3D points onto the image. Do it only for RGB images for now.
    if isinstance(images, np.ndarray):
        assert isinstance(trace_pts_list_or_dict, list), "trace_pts must be a list if images is a single image"
        images = draw_projected_trajectory_if_rgb(images, intrinsics, trace_pts_list_or_dict)

    elif isinstance(images, dict):
        for sub_path, image in images.items():
            trace_pts_list = trace_pts_list_or_dict
            if isinstance(trace_pts_list_or_dict, dict) and sub_path in trace_pts_list_or_dict:
                trace_pts_list = trace_pts_list_or_dict[sub_path]
            else:
                assert isinstance(trace_pts_list_or_dict, list), (
                    "trace_pts must be a list if images is a dict and sub_path not in trace_pts"
                )
            intrinsics_for_image = intrinsics
            if isinstance(intrinsics, dict) and sub_path in intrinsics:
                intrinsics_for_image = intrinsics[sub_path]
            images[sub_path] = draw_projected_trajectory_if_rgb(image, intrinsics_for_image, trace_pts_list)
    return images
