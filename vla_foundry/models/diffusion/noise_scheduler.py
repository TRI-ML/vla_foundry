import torch
import torch.nn as nn
import torch.nn.functional as F

from vla_foundry.params.model_params import NoiseSchedulerParams


class NoiseScheduler:
    def __init__(self, params: NoiseSchedulerParams):
        pass

    def add_noise(self, x_start, noise, timesteps):
        raise NotImplementedError

    def step(self, model_output, timestep, sample):
        raise NotImplementedError


class NoiseSchedulerDDPM(nn.Module, NoiseScheduler):
    def __init__(self, params: NoiseSchedulerParams):
        nn.Module.__init__(self)
        NoiseScheduler.__init__(self, params)
        self.num_timesteps = params.num_timesteps
        self.beta_start = params.beta_start
        self.beta_end = params.beta_end
        self.clamp_range = params.clamp_range
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer("posterior_variance", betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

    def add_noise(self, x_start, noise, timesteps, mask=None):
        # x_start, noise shape [bsz, channels, h, w]
        # self.sqrt_alphas_cumprod[timesteps] shape [bsz]

        # Determine how many dimensions to add by comparing timesteps and x_start shapes
        # timesteps is typically (batch_size,) and x_start is (batch_size, ..., feature)
        # We need to add dimensions to timesteps to match x_start's broadcasting requirements
        target_ndim = x_start.ndim
        current_ndim = timesteps.ndim
        dims_to_add = target_ndim - current_ndim

        # Create the view shape: keep first dimension (-1), add 1s for the remaining dimensions
        view_shape = [-1] + [1] * dims_to_add

        if mask is not None:
            # Ensure mask can broadcast with x_start by expanding missing dimensions
            # mask should have first dimensions matching x_start, and we'll expand the rest
            mask_expanded = mask
            while mask_expanded.ndim < x_start.ndim:
                mask_expanded = mask_expanded.unsqueeze(-1)
            mask_expanded = mask_expanded.to(dtype=x_start.dtype)
            # When mask=1: should behave exactly like no mask
            # When mask=0: should return original x_start (no noise)
            # Formula: mask * (normal_noisy_result) + (1 - mask) * x_start
            normal_result = (
                self.sqrt_alphas_cumprod[timesteps].view(*view_shape) * x_start
                + self.sqrt_one_minus_alphas_cumprod[timesteps].view(*view_shape) * noise
            ).to(dtype=x_start.dtype)
            output = mask_expanded * normal_result + (1 - mask_expanded) * x_start
        else:
            output = (
                self.sqrt_alphas_cumprod[timesteps].view(*view_shape) * x_start
                + self.sqrt_one_minus_alphas_cumprod[timesteps].view(*view_shape) * noise
            ).to(dtype=x_start.dtype)  # [bsz, channels, h, w]
        if self.clamp_range is not None:
            output = output.clamp(self.clamp_range[0], self.clamp_range[1])
        return output

    def step(self, model_output, timestep, sample):
        """Reverse process single step"""
        t = timestep

        # Get coefficients
        alpha_t = self.alphas[t]  # scalar
        alpha_cumprod_t = self.alphas_cumprod[t]  # scalar
        alpha_cumprod_t_prev = self.alphas_cumprod_prev[t]  # scalar
        beta_t = self.betas[t]  # scalar

        # Compute predicted original sample
        pred_original_sample = (sample - torch.sqrt(1 - alpha_cumprod_t) * model_output) / torch.sqrt(
            alpha_cumprod_t
        )  # [batch_size, channels, height, width]

        # Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = torch.sqrt(alpha_cumprod_t_prev) * beta_t / (1 - alpha_cumprod_t)  # scalar
        current_sample_coeff = torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)  # scalar

        # Compute predicted previous sample
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        )  # [batch_size, channels, height, width]

        # Add noise if not the last timestep
        if t > 0:
            noise = torch.randn_like(sample)  # [batch_size, channels, height, width]
            variance = torch.sqrt(self.posterior_variance[t]) * noise  # [batch_size, channels, height, width]
            pred_prev_sample = pred_prev_sample + variance  # [batch_size, channels, height, width]

        if self.clamp_range is not None:
            pred_prev_sample = pred_prev_sample.clamp(self.clamp_range[0], self.clamp_range[1])
        return pred_prev_sample  # [batch_size, channels, height, width]
