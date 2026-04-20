import pytest
import torch

from vla_foundry.models.diffusion.noise_scheduler import NoiseSchedulerDDPM
from vla_foundry.models.diffusion.noise_scheduler_diffusers import FlowMatchingScheduler, NoiseSchedulerDDPMDiffusers
from vla_foundry.params.model_params import NoiseSchedulerParams


class TestNoiseSchedulers:
    @pytest.fixture
    def noise_scheduler_params(self):
        return NoiseSchedulerParams(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)

    @pytest.fixture
    def clamped_noise_scheduler_params(self):
        return NoiseSchedulerParams(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, clamp_range=(-0.5, 0.5))

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        batch_size = 2
        channels = 3
        height = 32
        width = 32

        x_start = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(x_start)
        timesteps = torch.randint(0, 1000, (batch_size,))

        # Create a mask where first sample is fully masked (1), second is partially masked
        # Use 1D mask [batch] that can be expanded to match any tensor shape
        mask = torch.ones(batch_size, dtype=torch.float32)
        mask[1] = 0  # Second sample: fully mask out

        return x_start, noise, timesteps, mask

    @pytest.fixture
    def ddpm_scheduler(self, noise_scheduler_params):
        return NoiseSchedulerDDPM(noise_scheduler_params)

    @pytest.fixture
    def ddpm_diffusers_scheduler(self, noise_scheduler_params):
        return NoiseSchedulerDDPMDiffusers(noise_scheduler_params)

    @pytest.fixture
    def flow_matching_scheduler(self, noise_scheduler_params):
        return FlowMatchingScheduler(noise_scheduler_params)

    def test_ddpm_scheduler_add_noise_without_mask(self, ddpm_scheduler, sample_data):
        """Test DDPM scheduler add_noise without mask"""
        x_start, noise, timesteps, _ = sample_data

        noisy_x = ddpm_scheduler.add_noise(x_start, noise, timesteps)

        # Check output shape matches input
        assert noisy_x.shape == x_start.shape

        # Check that noise was actually added (output should be different from input)
        assert not torch.allclose(noisy_x, x_start)

        # Check that the output is deterministic given same inputs
        noisy_x2 = ddpm_scheduler.add_noise(x_start, noise, timesteps)
        assert torch.allclose(noisy_x, noisy_x2)

    def test_ddpm_scheduler_add_noise_with_mask(self, ddpm_scheduler, sample_data):
        """Test DDPM scheduler add_noise with mask"""
        x_start, noise, timesteps, mask = sample_data

        noisy_x = ddpm_scheduler.add_noise(x_start, noise, timesteps, mask=mask)
        noisy_x_no_mask = ddpm_scheduler.add_noise(x_start, noise, timesteps)

        # Check output shape matches input
        assert noisy_x.shape == x_start.shape

        # With mask [1., 0.], first sample (mask=1) should behave normally (same as no mask)
        # Second sample (mask=0) should be different due to masking
        assert torch.allclose(noisy_x[0], noisy_x_no_mask[0])  # First sample: mask=1, should be same as no mask
        assert not torch.allclose(noisy_x[1], noisy_x_no_mask[1])  # Second sample: mask=0, should be different

    def test_ddpm_scheduler_step(self, ddpm_scheduler, sample_data):
        """Test DDPM scheduler step method"""
        x_start, noise, timesteps, _ = sample_data

        # Create some sample model output (predicted noise)
        model_output = torch.randn_like(x_start)

        # Test step for different timesteps
        for t in [999, 500, 1, 0]:
            timestep = torch.tensor(t)
            sample = torch.randn_like(x_start)

            prev_sample = ddpm_scheduler.step(model_output, timestep, sample)

            # Check output shape matches input
            assert prev_sample.shape == sample.shape

            # For t=0, should not add noise (deterministic)
            if t == 0:
                prev_sample2 = ddpm_scheduler.step(model_output, timestep, sample)
                assert torch.allclose(prev_sample, prev_sample2)

    def test_ddpm_diffusers_scheduler_add_noise(self, ddpm_diffusers_scheduler, sample_data):
        """Test DDPM Diffusers scheduler add_noise method"""
        x_start, noise, timesteps, _ = sample_data

        noisy_x = ddpm_diffusers_scheduler.add_noise(x_start, noise, timesteps)

        # Check output shape matches input
        assert noisy_x.shape == x_start.shape

        # Check that noise was actually added
        assert not torch.allclose(noisy_x, x_start)

        # Check deterministic behavior
        noisy_x2 = ddpm_diffusers_scheduler.add_noise(x_start, noise, timesteps)
        assert torch.allclose(noisy_x, noisy_x2)

    def test_ddpm_diffusers_scheduler_step(self, ddpm_diffusers_scheduler, sample_data):
        """Test DDPM Diffusers scheduler step method"""
        x_start, noise, timesteps, _ = sample_data

        model_output = torch.randn_like(x_start)
        sample = torch.randn_like(x_start)
        timestep = torch.tensor(500)

        prev_sample = ddpm_diffusers_scheduler.step(model_output, timestep, sample)

        # Check output shape matches input
        assert prev_sample.shape == sample.shape

        # Check that step produces different output from input
        assert not torch.allclose(prev_sample, sample)

    def test_flow_matching_scheduler_add_noise_without_mask(self, flow_matching_scheduler, sample_data):
        """Test Flow Matching scheduler add_noise without mask"""
        x_start, noise, timesteps, _ = sample_data

        noisy_x = flow_matching_scheduler.add_noise(x_start, noise, timesteps)

        # Check output shape matches input
        assert noisy_x.shape == x_start.shape

        # Check that noise was actually added
        assert not torch.allclose(noisy_x, x_start)

        # Check deterministic behavior
        noisy_x2 = flow_matching_scheduler.add_noise(x_start, noise, timesteps)
        assert torch.allclose(noisy_x, noisy_x2)

    def test_flow_matching_scheduler_add_noise_with_mask(self, flow_matching_scheduler, sample_data):
        """Test Flow Matching scheduler add_noise with mask"""
        x_start, noise, timesteps, mask = sample_data

        noisy_x = flow_matching_scheduler.add_noise(x_start, noise, timesteps, mask=mask)
        noisy_x_no_mask = flow_matching_scheduler.add_noise(x_start, noise, timesteps)

        # Check output shape matches input
        assert noisy_x.shape == x_start.shape

        # Check that mask affects the output
        assert not torch.allclose(noisy_x, noisy_x_no_mask)

        # For masked regions (mask=0), should preserve the original signal (no noise)
        # For unmasked regions (mask=1), should be similar to no-mask version (noise added)
        assert torch.allclose(noisy_x[mask == 1], noisy_x_no_mask[mask == 1])
        assert not torch.allclose(noisy_x[mask == 0], noisy_x_no_mask[mask == 0])

    def test_flow_matching_scheduler_step(self, flow_matching_scheduler, sample_data):
        """Test Flow Matching scheduler step method"""
        x_start, noise, timesteps, _ = sample_data

        model_output = torch.randn_like(x_start)
        sample = torch.randn_like(x_start)
        timestep = torch.tensor(500)

        prev_sample = flow_matching_scheduler.step(model_output, timestep, sample)

        # Check output shape matches input
        assert prev_sample.shape == sample.shape

        # Test with different step sizes
        prev_sample_step2 = flow_matching_scheduler.step(model_output, timestep, sample, step_size=2)
        assert prev_sample_step2.shape == sample.shape

        # Different step sizes should produce different outputs
        assert not torch.allclose(prev_sample, prev_sample_step2)

    def test_mask_shapes_and_broadcasting(self, ddpm_scheduler, flow_matching_scheduler):
        """Test that masks work with different tensor shapes and automatically expand dimensions"""
        # Test with 4D input tensor and different VALID mask shapes
        batch_size, channels, height, width = 2, 3, 32, 32
        x_start = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(x_start)
        timesteps = torch.randint(0, 1000, (batch_size,))

        # Test different mask shapes where first dimensions match properly
        mask_shapes_to_test = [
            (batch_size,),  # 1D: [batch] -> expand to [batch, 1, 1, 1]
            (batch_size, 1),  # 2D: [batch, 1] -> expand to [batch, 1, 1, 1]
            (batch_size, 1, 1),  # 3D: [batch, 1, 1] -> expand to [batch, 1, 1, 1]
            (batch_size, channels, height, width),  # 4D: [batch, channels, height, width] -> no expansion needed
            (batch_size, 1, height, width),  # 4D: [batch, 1, height, width] -> broadcasts with channels
            (batch_size, channels, 1, width),  # 4D: [batch, channels, 1, width] -> broadcasts with height
            (batch_size, channels, height, 1),  # 4D: [batch, channels, height, 1] -> broadcasts with width
        ]

        for mask_shape in mask_shapes_to_test:
            mask = torch.ones(mask_shape, dtype=torch.float32)

            # Test DDPM scheduler with mask
            noisy_x_ddpm = ddpm_scheduler.add_noise(x_start, noise, timesteps, mask=mask)
            assert noisy_x_ddpm.shape == x_start.shape

            # Test Flow Matching scheduler with mask
            noisy_x_fm = flow_matching_scheduler.add_noise(x_start, noise, timesteps, mask=mask)
            assert noisy_x_fm.shape == x_start.shape

        # Test with 3D input tensor and valid mask shapes
        x_start_3d = torch.randn(2, 64, 64)
        noise_3d = torch.randn_like(x_start_3d)
        timesteps_3d = torch.randint(0, 1000, (2,))

        mask_shapes_3d = [
            (2,),  # 1D: [batch] -> expand to [batch, 1, 1]
            (2, 1),  # 2D: [batch, 1] -> expand to [batch, 1, 1]
            (2, 64, 64),  # 3D: [batch, height, width] -> no expansion needed
            (2, 1, 64),  # 3D: [batch, 1, width] -> broadcasts with height
            (2, 64, 1),  # 3D: [batch, height, 1] -> broadcasts with width
        ]

        for mask_shape in mask_shapes_3d:
            mask = torch.ones(mask_shape, dtype=torch.float32)

            # Test Flow Matching scheduler with 3D input
            noisy_x_fm = flow_matching_scheduler.add_noise(x_start_3d, noise_3d, timesteps_3d, mask=mask)
            assert noisy_x_fm.shape == x_start_3d.shape

    def test_timestep_edge_cases(self, ddpm_scheduler, flow_matching_scheduler):
        """Test schedulers with edge case timestep values"""
        x_start = torch.randn(2, 3, 32, 32)
        noise = torch.randn_like(x_start)

        # Test with timestep 0 (start of process)
        timesteps_start = torch.zeros(2, dtype=torch.long)
        noisy_x_start = ddpm_scheduler.add_noise(x_start, noise, timesteps_start)
        noisy_x_start_fm = flow_matching_scheduler.add_noise(x_start, noise, timesteps_start)

        # Test with max timestep (end of process)
        timesteps_end = torch.full((2,), 999, dtype=torch.long)
        noisy_x_end = ddpm_scheduler.add_noise(x_start, noise, timesteps_end)
        noisy_x_end_fm = flow_matching_scheduler.add_noise(x_start, noise, timesteps_end)

        # All should produce valid outputs
        assert noisy_x_start.shape == x_start.shape
        assert noisy_x_start_fm.shape == x_start.shape
        assert noisy_x_end.shape == x_start.shape
        assert noisy_x_end_fm.shape == x_start.shape

    def test_scheduler_dtype_consistency(self, ddpm_scheduler, flow_matching_scheduler):
        """Test that schedulers preserve input dtypes"""
        dtypes_to_test = [torch.float32, torch.float16]

        for dtype in dtypes_to_test:
            x_start = torch.randn(2, 3, 32, 32, dtype=dtype)
            noise = torch.randn_like(x_start)
            timesteps = torch.randint(0, 1000, (2,))

            # Test DDPM scheduler
            noisy_x_ddpm = ddpm_scheduler.add_noise(x_start, noise, timesteps)
            # Note: DDPM scheduler might promote to float32 due to internal computations
            assert noisy_x_ddpm.dtype == dtype, f"Expected {dtype}, got {noisy_x_ddpm.dtype}"

            # Test Flow Matching scheduler (should preserve dtype better)
            noisy_x_fm = flow_matching_scheduler.add_noise(x_start, noise, timesteps)
            assert noisy_x_fm.dtype == dtype, f"Expected {dtype}, got {noisy_x_fm.dtype}"

    def test_batch_size_consistency(self, ddpm_scheduler, flow_matching_scheduler):
        """Test schedulers with different batch sizes"""
        batch_sizes = [1, 4, 8]

        for batch_size in batch_sizes:
            x_start = torch.randn(batch_size, 3, 32, 32)
            noise = torch.randn_like(x_start)
            timesteps = torch.randint(0, 1000, (batch_size,))

            # Test all schedulers
            noisy_x_ddpm = ddpm_scheduler.add_noise(x_start, noise, timesteps)
            noisy_x_fm = flow_matching_scheduler.add_noise(x_start, noise, timesteps)

            assert noisy_x_ddpm.shape == x_start.shape
            assert noisy_x_fm.shape == x_start.shape
            assert noisy_x_ddpm.shape[0] == batch_size
            assert noisy_x_fm.shape[0] == batch_size

    @pytest.mark.parametrize(
        "scheduler_cls",
        [NoiseSchedulerDDPM, NoiseSchedulerDDPMDiffusers, FlowMatchingScheduler],
    )
    def test_clamp_range_limits_outputs(self, scheduler_cls, clamped_noise_scheduler_params):
        """Ensure clamp_range enforces bounds on add_noise and step outputs"""
        scheduler = scheduler_cls(clamped_noise_scheduler_params)
        clamp_min, clamp_max = clamped_noise_scheduler_params.clamp_range

        x_start = torch.full((1, 3, 4, 4), 10.0)
        noise = torch.full_like(x_start, -10.0)
        timesteps = torch.zeros(1, dtype=torch.long)

        noisy_x = scheduler.add_noise(x_start, noise, timesteps)
        assert torch.all(noisy_x <= clamp_max)
        assert torch.all(noisy_x >= clamp_min)

        model_output = torch.full_like(x_start, -10.0)
        sample = torch.full_like(x_start, 10.0)
        timestep = torch.tensor(0)

        if isinstance(scheduler, FlowMatchingScheduler):
            prev_sample = scheduler.step(model_output, timestep, sample, step_size=1)
        else:
            prev_sample = scheduler.step(model_output, timestep, sample)

        assert torch.all(prev_sample <= clamp_max)
        assert torch.all(prev_sample >= clamp_min)
