"""Diffusion models including Stable Diffusion and noise schedulers."""

from vla_foundry.models.diffusion.noise_scheduler import NoiseSchedulerDDPM
from vla_foundry.models.diffusion.noise_scheduler_diffusers import (
    FlowMatchingScheduler,
    NoiseSchedulerDDPMDiffusers,
)
from vla_foundry.models.diffusion.stable_diffusion import StableDiffusion
from vla_foundry.models.diffusion.unet import (
    CrossAttentionBlock,
    ResnetBlock,
    SelfAttentionBlock,
    UNet,
)
from vla_foundry.models.diffusion.unet_diffusers import UNetDiffusers
from vla_foundry.models.registry import create_model, register_model
from vla_foundry.params.model_params import ModelParams


def create_noise_scheduler(model_params):
    """Create noise scheduler based on model parameters."""
    if model_params.use_diffusers_scheduler:
        noise_scheduler = NoiseSchedulerDDPMDiffusers(model_params.noise_scheduler)
    elif model_params.use_flow_matching_scheduler:
        noise_scheduler = FlowMatchingScheduler(model_params.noise_scheduler)
    else:
        noise_scheduler = NoiseSchedulerDDPM(model_params.noise_scheduler)
    return noise_scheduler


@register_model("stable_diffusion")
def create_stable_diffusion(model_params: ModelParams, load_pretrained: bool = True):
    clip = create_model(model_params.clip, load_pretrained) if model_params.clip.hf_pretrained is not None else None
    unet = UNetDiffusers(model_params.unet) if model_params.use_diffusers_unet else UNet(model_params.unet)
    noise_scheduler = create_noise_scheduler(model_params)
    return StableDiffusion(model_params, clip, unet, noise_scheduler)


__all__ = [
    "NoiseSchedulerDDPM",
    "NoiseSchedulerDDPMDiffusers",
    "FlowMatchingScheduler",
    "StableDiffusion",
    "UNet",
    "UNetDiffusers",
    "ResnetBlock",
    "SelfAttentionBlock",
    "CrossAttentionBlock",
    "create_noise_scheduler",
]
