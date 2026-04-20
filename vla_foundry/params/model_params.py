import logging
from dataclasses import dataclass, field
from typing import Literal

import draccus

from vla_foundry.data.processor import get_processor
from vla_foundry.params.base_params import BaseParams


def register_model_params(key: str):
    """
    Registers a ModelParams subclass and sets its type attribute.
    Use decorator wrapper because draccus's model selection with --model.type doesn't
    automatically populate the attribute cfg.model.type
    """

    def decorator(cls):
        registered_cls = ModelParams.register_subclass(key)(cls)
        registered_cls._type = key
        return registered_cls

    return decorator


@dataclass(frozen=True)
class ModelParams(draccus.ChoiceRegistry, BaseParams):
    type: str = field(default=None)
    resume_from_checkpoint: str = field(default=None)
    resume_weights_only: bool = field(default=False)
    freeze: bool = field(default=False)

    def __init__(self):
        raise NotImplementedError("ModelParams should not be instantiated directly. Use a subclass with model.type=...")

    def __post_init__(self):
        super().__post_init__()
        if self.type is None:
            object.__setattr__(self, "type", getattr(self, "_type", None))


@register_model_params("transformer")
@dataclass(frozen=True)
class TransformerParams(ModelParams):
    norm_type: str = field(default="default_layer_norm")
    ffn_type: str = field(default="swiglu")
    qk_norm: bool = field(default=False)
    positional_embedding_type: str = field(default="rotary")
    attn_name: str = field(default="torch_attn")
    hidden_dim: int = field(default=96)
    n_layers: int = field(default=8)
    n_heads: int = field(default=4)
    vocab_size: int = field(default=50432)
    post_embed_norm: bool = field(default=False)
    norm_eps: float = field(default=1e-5)
    weight_tying: bool = field(default=False)
    cast_output_to_float32: bool = field(default=False)
    max_seq_len: int = field(default=2048)
    is_causal: bool = field(default=True)


@register_model_params("transformer_hf")
@dataclass(frozen=True)
class TransformerHFParams(ModelParams):
    hf_pretrained: str = field(default=None)
    _hf_config = None

    @property
    def hidden_dim(self):
        if self._hf_config is None:
            from transformers import AutoConfig

            object.__setattr__(self, "_hf_config", AutoConfig.from_pretrained(self.hf_pretrained))
        return self._hf_config.hidden_size

    @property
    def vocab_size(self):
        if self._hf_config is None:
            from transformers import AutoConfig

            object.__setattr__(self, "_hf_config", AutoConfig.from_pretrained(self.hf_pretrained))
        return self._hf_config.vocab_size


@register_model_params("vit")
@dataclass(frozen=True)
class ViTParams(ModelParams):
    pretrained: str = field(default=None)
    interpolation_mode: str = field(default="bicubic")
    hidden_dim: int = field(default=768)
    inter_dim: int = field(default=3072)
    patch_size: int = field(default=16)
    img_size: int = field(default=384)
    n_heads: int = field(default=12)
    dropout: float = field(default=0.0)
    n_layers: int = field(default=12)
    ln_eps: float = field(default=1e-6)
    cls_flag: bool = field(default=False)
    projector_pixel_shuffle_factor: int = field(default=1)


@register_model_params("vit_hf")
@dataclass(frozen=True)
class ViTHFParams(TransformerHFParams):
    hidden_dim: int = field(default=768)
    projector_pixel_shuffle_factor: int = field(default=1)


@register_model_params("vlm")
@dataclass(frozen=True)
class VLMParams(ModelParams):
    vit: ViTParams | ViTHFParams = field(default_factory=ViTParams)
    transformer: TransformerParams | TransformerHFParams = field(default_factory=TransformerParams)
    image_token_id: int = field(default=None)
    processor: str = field(default=None)

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        if self.processor is None and hasattr(cfg.data, "processor") and cfg.data.processor is not None:
            object.__setattr__(self, "processor", cfg.data.processor)

        # Prefer computing special ids from the processor/tokenizer rather than requiring user input
        # 1) Resolve processor once (reuse if already loaded on data params)
        processor = getattr(cfg.data, "processor_loaded", None)
        if processor is None and hasattr(cfg.data, "processor") and cfg.data.processor is not None:
            processor = get_processor(cfg.data)

        # 2) Compute image_token_id if available from data or processor
        image_token_id = getattr(cfg.data, "image_token_id", None)
        if image_token_id is None and processor is not None:
            image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is not None and getattr(self, "image_token_id", None) is None:
            object.__setattr__(self, "image_token_id", image_token_id)

        # 3) Compute vocab_size from tokenizer length if transformer params exist and not explicitly overridden
        if hasattr(self, "transformer") and hasattr(self.transformer, "vocab_size") and processor is not None:
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                computed_vocab_size = None
                # Prefer __len__ if available
                if hasattr(tokenizer, "__len__"):
                    length_value = len(tokenizer)
                    if isinstance(length_value, int) and length_value > 0:
                        computed_vocab_size = length_value
                # Fallback to get_vocab if available
                if computed_vocab_size is None and hasattr(tokenizer, "get_vocab"):
                    vocab = tokenizer.get_vocab()
                    if isinstance(vocab, dict) and len(vocab) > 0:
                        computed_vocab_size = len(vocab)
                # Fallback to attribute vocab_size if available
                if computed_vocab_size is None and hasattr(tokenizer, "vocab_size"):
                    vs = tokenizer.vocab_size
                    if isinstance(vs, int) and vs > 0:
                        computed_vocab_size = vs

                if isinstance(computed_vocab_size, int) and computed_vocab_size > 0:
                    current_vocab_size = getattr(self.transformer, "vocab_size", None)
                    # Only override when it's the default/sentinel value
                    if current_vocab_size in (None, 0, TransformerParams.vocab_size):
                        object.__setattr__(self.transformer, "vocab_size", computed_vocab_size)


@register_model_params("vlm_hf")
@dataclass(frozen=True)
class VLMHFParams(TransformerHFParams):
    pass


@register_model_params("unet")
@dataclass(frozen=True)
class UNetParams(ModelParams):
    in_channels: int = field(default=3)
    out_channels: int = field(default=3)
    time_emb_dim: int = field(default=256)
    text_emb_dim: int = field(default=512)
    channels: list[int] = field(default_factory=list)
    image_size: int = field(default=128)
    time_mlp_float32: bool = field(default=False)


@register_model_params("noise_scheduler")
@dataclass(frozen=True)
class NoiseSchedulerParams(ModelParams):
    num_timesteps: int = field(default=1000)
    beta_start: float = field(default=0.0001)
    beta_end: float = field(default=0.02)
    clamp_range: tuple[float, float] = field(default=(-1.5, 1.5))

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        if hasattr(cfg.data, "normalization") and self.clamp_range is not None:
            if not cfg.data.normalization.enabled:
                logging.warning(
                    "Normalization is disabled with clamping range enabled. "
                    f"Make sure your data is within the clamping range. {self.clamp_range}"
                )
            elif not cfg.data.normalization.centered_norm and self.clamp_range[0] == -self.clamp_range[1]:
                raise ValueError(
                    f"Clamp range {self.clamp_range} is symetric but "
                    f"normalization is not centered: {cfg.data.normalization.centered_norm}"
                    f"Set data.normalization.centered_norm to True or use a different clamp range."
                )


@register_model_params("clip_hf")
@dataclass(frozen=True)
class CLIPHFParams(TransformerHFParams):
    freeze_text_encoder: bool = field(default=False)
    freeze_image_encoder: bool = field(default=False)


@register_model_params("clip_openclip")
@dataclass(frozen=True)
class CLIP_OpenCLIPParams(ModelParams):
    architecture: str = field(default=None)
    pretrained_weights: str = field(default=None)
    freeze_text_encoder: bool = field(default=False)
    freeze_image_encoder: bool = field(default=False)


@dataclass(frozen=True)
class BackboneParams(ModelParams):
    """Marker base class for vision-language backbone configurations."""

    pass


@register_model_params("clip_backbone")
@dataclass(frozen=True)
class CLIPBackboneParams(BackboneParams, CLIPHFParams):
    disable_text: bool = field(default=False)


@register_model_params("vlm_backbone")
@dataclass(frozen=True)
class VLMBackboneParams(BackboneParams, VLMHFParams):
    # Number of last VLM layers to extract hidden states from for diffusion
    num_vlm_layers_to_use: int = field(default=4)


@register_model_params("vlm_foundry_backbone")
@dataclass(frozen=True)
class VLMFoundryBackboneParams(BackboneParams):
    # Number of last VLM layers to extract hidden states from for diffusion
    num_vlm_layers_to_use: int = field(default=4)
    # Training-time pointer to source VLM experiment (unused at inference)
    vlm_experiment_dir: str | None = field(default=None)


@register_model_params("vit_backbone")
@dataclass(frozen=True)
class ViTBackboneParams(BackboneParams, ViTParams):
    pass


@register_model_params("stable_diffusion")
@dataclass(frozen=True)
class StableDiffusionParams(ModelParams):
    unet: UNetParams = field(default_factory=UNetParams)
    noise_scheduler: NoiseSchedulerParams = field(default_factory=NoiseSchedulerParams)

    use_diffusers_unet: bool = field(default=False)
    use_diffusers_scheduler: bool = field(default=False)
    use_flow_matching_scheduler: bool = field(default=False)

    clip: CLIPHFParams = field(default_factory=CLIPHFParams)

    # CFG params
    do_classifier_free_guidance: bool = field(default=False)
    guidance_scale: float = field(default=4.0)  # Standard CFG scale
    dropout_percent: float = field(default=0.2)  # 20%% dropout for unconditional training

    @property
    def image_size(self):
        return self.unet.image_size


@register_model_params("diffusion_policy")
@dataclass(frozen=True)
class DiffusionPolicyParams(ModelParams):
    vision_language_backbone: VLMBackboneParams | CLIPBackboneParams | VLMFoundryBackboneParams | ViTBackboneParams = (
        field(default_factory=CLIPBackboneParams)
    )
    transformer: TransformerParams | TransformerHFParams = field(default_factory=ModelParams)
    noise_scheduler: NoiseSchedulerParams = field(default_factory=NoiseSchedulerParams)

    use_diffusers_scheduler: bool = field(default=False)
    use_flow_matching_scheduler: bool = field(default=False)
    input_noise_std: float = field(default=0.0)
    diffusion_step_conditioning: Literal["add", "concat"] = field(default="concat")
    num_action_head_repeats: int = field(default=None)

    # Shared attributes. Overwritten in init_shared_attributes.
    action_dim: int = field(default=None)
    proprioception_dim: int = field(default=0)

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        object.__setattr__(self, "action_dim", cfg.data.action_dim)
        object.__setattr__(self, "proprioception_dim", cfg.data.proprioception_dim)
