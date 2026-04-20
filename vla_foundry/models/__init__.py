"""
Models module with registry-based model creation.

This module imports all model definitions, which register themselves via decorators.
New models should be registered in their respective files or subdirectories.
"""

# Import registry functions first
# Model subdirectories
# Import batch handlers module to trigger registration
import vla_foundry.models.batch_handlers
import vla_foundry.models.diffusion  # registers "stable_diffusion"
import vla_foundry.models.diffusion_policy  # registers "diffusion_policy", "clip_hf", "clip_openclip"

# Import all model modules to trigger their registrations
# Individual model files
import vla_foundry.models.transformer  # registers "transformer"
import vla_foundry.models.transformer_hf  # registers "transformer_hf"
import vla_foundry.models.vlm  # registers "vlm"
import vla_foundry.models.vlm_hf  # registers "vlm_hf"

# Import helper function
from vla_foundry.models.diffusion import create_noise_scheduler

# Import registry functions
from vla_foundry.models.registry import create_batch_handler, create_model, register_model
