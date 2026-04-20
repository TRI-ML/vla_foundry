import numpy as np
import pytest

from vla_foundry.data.preprocessing.image_utils import ImageResizingMethod
from vla_foundry.data.robotics.cv_utils import scale_intrinsics_4_for_resize_and_crop


def assert_allclose(actual, expected, rtol=1e-6, atol=1e-6):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol)


class TestScaleIntrinsics4ForResizeAndCrop:
    """Tests for scale_intrinsics_4_for_resize_and_crop (fx, fy, cx, cy)."""

    @pytest.fixture
    def sample_intrinsics_4(self):
        # fx, fy, cx, cy
        return np.array([800.0, 800.0, 320.0, 240.0], dtype=np.float64)

    @pytest.fixture
    def batch_intrinsics_4(self):
        return np.array(
            [
                [800.0, 800.0, 320.0, 240.0],
                [1000.0, 1000.0, 512.0, 384.0],
            ],
            dtype=np.float64,
        )

    # -------------------------
    # RESIZE_NO_CROP (distort)
    # -------------------------
    def test_resize_no_crop_single(self, sample_intrinsics_4):
        original_size = (640, 480)
        processed_size = (320, 240)  # exact 0.5 in both

        result = scale_intrinsics_4_for_resize_and_crop(
            sample_intrinsics_4,
            original_size,
            processed_size,
            resize_method=ImageResizingMethod.RESIZE_NO_CROP,
        )

        expected = np.array([400.0, 400.0, 160.0, 120.0], dtype=np.float64)
        assert_allclose(result, expected)

    def test_resize_no_crop_batch(self, batch_intrinsics_4):
        original_size = (1024, 768)
        processed_size = (512, 384)  # exact 0.5 in both

        result = scale_intrinsics_4_for_resize_and_crop(
            batch_intrinsics_4,
            original_size,
            processed_size,
            resize_method=ImageResizingMethod.RESIZE_NO_CROP,
        )

        expected = np.array(
            [
                [400.0, 400.0, 160.0, 120.0],
                [500.0, 500.0, 256.0, 192.0],
            ],
            dtype=np.float64,
        )
        assert_allclose(result, expected)

    def test_resize_no_crop_anisotropic(self, sample_intrinsics_4):
        # Different scale factors in x and y
        original_size = (640, 480)
        processed_size = (320, 480)  # sx=0.5, sy=1.0

        result = scale_intrinsics_4_for_resize_and_crop(
            sample_intrinsics_4,
            original_size,
            processed_size,
            resize_method=ImageResizingMethod.RESIZE_NO_CROP,
        )

        expected = np.array(
            [
                800.0 * 0.5,  # fx
                800.0 * 1.0,  # fy
                320.0 * 0.5,  # cx
                240.0 * 1.0,  # cy
            ],
            dtype=np.float64,
        )
        assert_allclose(result, expected)

    # -------------------------
    # CENTER_CROP (cover + crop)
    # -------------------------
    def test_center_crop_square_from_landscape(self, sample_intrinsics_4):
        # original 640x480 -> target 480x480
        # scale = max(480/640=0.75, 480/480=1.0) = 1.0
        # new_W=640,new_H=480; cx_offset=(640-480)/2=80; cy_offset=(480-480)/2=0
        original_size = (640, 480)
        processed_size = (480, 480)

        result = scale_intrinsics_4_for_resize_and_crop(
            sample_intrinsics_4,
            original_size,
            processed_size,
            resize_method=ImageResizingMethod.CENTER_CROP,
        )

        expected = np.array(
            [
                800.0 * 1.0,  # fx
                800.0 * 1.0,  # fy
                320.0 * 1.0 - 80.0,  # cx
                240.0 * 1.0 - 0.0,  # cy
            ],
            dtype=np.float64,
        )
        assert_allclose(result, expected)

    def test_center_crop_square_smaller(self, sample_intrinsics_4):
        # original 640x480 -> target 320x320
        # scale = max(320/640=0.5, 320/480=0.666666...) = 2/3
        # new_W=640*(2/3)=426.666..., new_H=480*(2/3)=320
        # cx_offset=(426.666...-320)/2=53.333..., cy_offset=(320-320)/2=0
        original_size = (640, 480)
        processed_size = (320, 320)

        scale = 320.0 / 480.0  # 2/3
        new_W = 640.0 * scale
        new_H = 480.0 * scale
        cx_offset = (new_W - 320.0) / 2.0
        cy_offset = (new_H - 320.0) / 2.0

        result = scale_intrinsics_4_for_resize_and_crop(
            sample_intrinsics_4,
            original_size,
            processed_size,
            resize_method=ImageResizingMethod.CENTER_CROP,
        )

        expected = np.array(
            [
                800.0 * scale,
                800.0 * scale,
                320.0 * scale - cx_offset,
                240.0 * scale - cy_offset,
            ],
            dtype=np.float64,
        )
        assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_center_crop_batch(self, batch_intrinsics_4):
        original_size = (640, 480)
        processed_size = (480, 480)

        # scale = 1.0; cx_offset=80; cy_offset=0
        result = scale_intrinsics_4_for_resize_and_crop(
            batch_intrinsics_4,
            original_size,
            processed_size,
            resize_method=ImageResizingMethod.CENTER_CROP,
        )

        expected = np.array(
            [
                [800.0, 800.0, 320.0 - 80.0, 240.0],
                [1000.0, 1000.0, 512.0 - 80.0, 384.0],
            ],
            dtype=np.float64,
        )
        assert_allclose(result, expected)

    # -------------------------
    # RESIZE_FIT (fit + pad)
    # -------------------------
    def test_resize_fit_letterbox_padding(self, sample_intrinsics_4):
        # original 640x480 -> target 640x640 (pad top/bottom)
        # scale = min(640/640=1, 640/480=1.333...) = 1
        # new_W=640, new_H=480
        # pad_x=(640-640)/2=0, pad_y=(640-480)/2=80
        original_size = (640, 480)
        processed_size = (640, 640)

        result = scale_intrinsics_4_for_resize_and_crop(
            sample_intrinsics_4,
            original_size,
            processed_size,
            resize_method=ImageResizingMethod.RESIZE_FIT,
        )

        expected = np.array(
            [
                800.0 * 1.0,
                800.0 * 1.0,
                320.0 * 1.0 + 0.0,
                240.0 * 1.0 + 80.0,
            ],
            dtype=np.float64,
        )
        assert_allclose(result, expected)

    def test_resize_fit_pillarbox_padding(self, sample_intrinsics_4):
        # original 640x480 -> target 800x480 (pad left/right)
        # scale=min(800/640=1.25, 480/480=1) = 1
        # new_W=640, new_H=480
        # pad_x=(800-640)/2=80, pad_y=0
        original_size = (640, 480)
        processed_size = (800, 480)

        result = scale_intrinsics_4_for_resize_and_crop(
            sample_intrinsics_4,
            original_size,
            processed_size,
            resize_method=ImageResizingMethod.RESIZE_FIT,
        )

        expected = np.array(
            [
                800.0,
                800.0,
                320.0 + 80.0,
                240.0,
            ],
            dtype=np.float64,
        )
        assert_allclose(result, expected)

    def test_resize_fit_downscale_with_padding(self, sample_intrinsics_4):
        # original 640x480 -> target 320x320 (fit inside -> pad y)
        # scale = min(320/640=0.5, 320/480=0.666...) = 0.5
        # new_W=320, new_H=240
        # pad_x=0, pad_y=(320-240)/2=40
        original_size = (640, 480)
        processed_size = (320, 320)

        scale = 0.5
        pad_x = 0.0
        pad_y = (320.0 - 240.0) / 2.0  # 40

        result = scale_intrinsics_4_for_resize_and_crop(
            sample_intrinsics_4,
            original_size,
            processed_size,
            resize_method=ImageResizingMethod.RESIZE_FIT,
        )

        expected = np.array(
            [
                800.0 * scale,
                800.0 * scale,
                320.0 * scale + pad_x,
                240.0 * scale + pad_y,
            ],
            dtype=np.float64,
        )
        assert_allclose(result, expected)

    # -------------------------
    # Basic properties / edge cases
    # -------------------------
    def test_same_size_identity_for_all_methods(self, sample_intrinsics_4):
        original_size = (640, 480)
        processed_size = (640, 480)

        for method in [
            ImageResizingMethod.CENTER_CROP,
            ImageResizingMethod.RESIZE_NO_CROP,
            ImageResizingMethod.RESIZE_FIT,
        ]:
            result = scale_intrinsics_4_for_resize_and_crop(
                sample_intrinsics_4,
                original_size,
                processed_size,
                resize_method=method,
            )
            assert_allclose(result, sample_intrinsics_4)

    def test_output_shape_single(self, sample_intrinsics_4):
        result = scale_intrinsics_4_for_resize_and_crop(
            sample_intrinsics_4, (640, 480), (320, 240), resize_method=ImageResizingMethod.RESIZE_NO_CROP
        )
        assert np.asarray(result).shape == (4,)

    def test_output_shape_batch(self, batch_intrinsics_4):
        result = scale_intrinsics_4_for_resize_and_crop(
            batch_intrinsics_4, (640, 480), (320, 240), resize_method=ImageResizingMethod.RESIZE_NO_CROP
        )
        assert np.asarray(result).shape == (2, 4)

    def test_invalid_resize_method_raises(self, sample_intrinsics_4):
        with pytest.raises(ValueError):
            scale_intrinsics_4_for_resize_and_crop(
                sample_intrinsics_4,
                (640, 480),
                (320, 240),
                resize_method="invalid_method",
            )

    def test_wrong_intrinsics_ndim_raises(self):
        intrinsics = np.zeros((2, 2, 4), dtype=np.float64)  # ndim=3
        with pytest.raises(ValueError):
            scale_intrinsics_4_for_resize_and_crop(
                intrinsics,
                (640, 480),
                (320, 240),
                resize_method=ImageResizingMethod.RESIZE_NO_CROP,
            )

    def test_wrong_intrinsics_last_dim_raises(self):
        intrinsics = np.zeros((3,), dtype=np.float64)  # last dim != 4
        with pytest.raises(ValueError):
            scale_intrinsics_4_for_resize_and_crop(
                intrinsics,
                (640, 480),
                (320, 240),
                resize_method=ImageResizingMethod.RESIZE_NO_CROP,
            )

        intrinsics2 = np.zeros((2, 5), dtype=np.float64)  # last dim != 4
        with pytest.raises(ValueError):
            scale_intrinsics_4_for_resize_and_crop(
                intrinsics2,
                (640, 480),
                (320, 240),
                resize_method=ImageResizingMethod.RESIZE_NO_CROP,
            )

    def test_zero_or_negative_sizes_do_not_nan(self, sample_intrinsics_4):
        # Your function doesn't explicitly validate sizes; at minimum ensure it doesn't produce NaNs/Infs
        # (If you later add validation, change this test to expect ValueError.)
        for processed_size in [(0, 240), (320, 0), (-320, 240), (320, -240)]:
            result = scale_intrinsics_4_for_resize_and_crop(
                sample_intrinsics_4,
                (640, 480),
                processed_size,
                resize_method=ImageResizingMethod.RESIZE_NO_CROP,
            )
            assert np.isfinite(np.asarray(result)).all()

    def test_batch_consistency(self, batch_intrinsics_4):
        original_size = (640, 480)
        processed_size = (320, 240)

        batch_result = scale_intrinsics_4_for_resize_and_crop(
            batch_intrinsics_4,
            original_size,
            processed_size,
            resize_method=ImageResizingMethod.RESIZE_NO_CROP,
        )

        indiv = []
        for i in range(batch_intrinsics_4.shape[0]):
            indiv.append(
                scale_intrinsics_4_for_resize_and_crop(
                    batch_intrinsics_4[i],
                    original_size,
                    processed_size,
                    resize_method=ImageResizingMethod.RESIZE_NO_CROP,
                )
            )
        indiv = np.stack(indiv, axis=0)

        assert_allclose(batch_result, indiv)

    def test_dtype_is_float(self, sample_intrinsics_4):
        for dtype in [np.float32, np.float64]:
            intr = sample_intrinsics_4.astype(dtype)
            result = scale_intrinsics_4_for_resize_and_crop(
                intr, (640, 480), (320, 240), resize_method=ImageResizingMethod.RESIZE_FIT
            )
            assert np.issubdtype(np.asarray(result).dtype, np.floating)

    def test_numerical_precision_exact_half_scale(self):
        intrinsics = np.array([1000.0, 1000.0, 500.0, 400.0], dtype=np.float64)
        original_size = (1000, 800)
        processed_size = (500, 400)  # exact 0.5 for both axes

        result = scale_intrinsics_4_for_resize_and_crop(
            intrinsics,
            original_size,
            processed_size,
            resize_method=ImageResizingMethod.RESIZE_NO_CROP,
        )

        expected = np.array([500.0, 500.0, 250.0, 200.0], dtype=np.float64)
        assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
