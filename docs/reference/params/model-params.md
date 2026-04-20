# ModelParams

`ModelParams` is the polymorphic base class for all model configurations. It uses draccus `ChoiceRegistry` so that the concrete subclass is selected at runtime based on the `type` field.

**Source:** `vla_foundry/params/model_params.py`

## Base Fields

Every `ModelParams` subclass inherits these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `None` | Registry key that selects the concrete subclass. Auto-populated from the registered key. |
| `resume_from_checkpoint` | `str` | `None` | Path to a checkpoint file to resume from. |
| `resume_weights_only` | `bool` | `False` | If `True`, load only model weights (skip optimizer state, step counter, etc.). |
| `freeze` | `bool` | `False` | If `True`, freeze all parameters in this module (no gradient updates). |

---

## TransformerParams

**Type key:** `transformer`

A from-scratch causal transformer (GPT-style) with configurable normalization, FFN type, and positional embeddings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `norm_type` | `str` | `"default_layer_norm"` | Normalization layer type. |
| `ffn_type` | `str` | `"swiglu"` | Feed-forward network type. |
| `qk_norm` | `bool` | `False` | Apply normalization to query and key projections. |
| `positional_embedding_type` | `str` | `"rotary"` | Positional embedding type (e.g., `"rotary"`, `"learned"`). |
| `attn_name` | `str` | `"torch_attn"` | Attention implementation. |
| `hidden_dim` | `int` | `96` | Model hidden dimension. |
| `n_layers` | `int` | `8` | Number of transformer layers. |
| `n_heads` | `int` | `4` | Number of attention heads. |
| `vocab_size` | `int` | `50432` | Vocabulary size for the embedding layer. |
| `post_embed_norm` | `bool` | `False` | Apply layer norm after the embedding layer. |
| `norm_eps` | `float` | `1e-5` | Epsilon for normalization layers. |
| `weight_tying` | `bool` | `False` | Tie input embedding and output projection weights. |
| `cast_output_to_float32` | `bool` | `False` | Cast the output logits to float32 (useful for mixed-precision stability). |
| `max_seq_len` | `int` | `2048` | Maximum sequence length. |
| `is_causal` | `bool` | `True` | Use causal (autoregressive) attention masking. |

---

## TransformerHFParams

**Type key:** `transformer_hf`

Loads a transformer from a Hugging Face pretrained checkpoint. The `hidden_dim` and `vocab_size` are dynamically read from the HF config.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hf_pretrained` | `str` | `None` | Hugging Face model identifier (e.g., `"Qwen/Qwen2.5-0.5B"`). |

**Dynamic properties:**

| Property | Source | Description |
|----------|--------|-------------|
| `hidden_dim` | `AutoConfig.hidden_size` | Retrieved from the HF model config on first access. |
| `vocab_size` | `AutoConfig.vocab_size` | Retrieved from the HF model config on first access. |

---

## ViTParams

**Type key:** `vit`

A from-scratch Vision Transformer for image encoding.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pretrained` | `str` | `None` | Path or identifier for pretrained ViT weights. |
| `interpolation_mode` | `str` | `"bicubic"` | Interpolation mode for positional embedding resizing. |
| `hidden_dim` | `int` | `768` | Hidden dimension. |
| `inter_dim` | `int` | `3072` | Intermediate (FFN) dimension. |
| `patch_size` | `int` | `16` | Patch size in pixels. |
| `img_size` | `int` | `384` | Input image size in pixels. |
| `n_heads` | `int` | `12` | Number of attention heads. |
| `dropout` | `float` | `0.0` | Dropout rate. |
| `n_layers` | `int` | `12` | Number of transformer layers. |
| `ln_eps` | `float` | `1e-6` | Layer norm epsilon. |
| `cls_flag` | `bool` | `False` | Include a CLS token. |
| `projector_pixel_shuffle_factor` | `int` | `1` | Pixel shuffle factor for the output projector. A factor of 2 reduces the token count by 4x. |

---

## ViTHFParams

**Type key:** `vit_hf`

A Hugging Face-backed ViT. Inherits from `TransformerHFParams`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hf_pretrained` | `str` | `None` | Hugging Face model identifier. *(Inherited from TransformerHFParams)* |
| `hidden_dim` | `int` | `768` | Hidden dimension (overrides the dynamic property from `TransformerHFParams`). |
| `projector_pixel_shuffle_factor` | `int` | `1` | Pixel shuffle factor for the output projector. |

---

## VLMParams

**Type key:** `vlm`

A Vision-Language Model that composes a ViT image encoder with a transformer language model.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vit` | `Union[ViTParams, ViTHFParams]` | `ViTParams()` | Image encoder configuration. |
| `transformer` | `Union[TransformerParams, TransformerHFParams]` | `TransformerParams()` | Language model configuration. |
| `image_token_id` | `int` | `None` | Token ID for image placeholder tokens. Auto-derived from the processor if not set. |
| `processor` | `str` | `None` | HF processor identifier (e.g., `"google/paligemma-3b-pt-224"`). Inherited from `DataParams.processor` if not set. |

!!! note "Shared Attributes"
    `VLMParams.init_shared_attributes` automatically resolves `image_token_id` and `vocab_size` from the data processor/tokenizer when possible, so you rarely need to set these manually.

---

## VLMHFParams

**Type key:** `vlm_hf`

Loads an entire VLM from a single Hugging Face checkpoint (e.g., PaliGemma, Qwen-VL).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hf_pretrained` | `str` | `None` | Hugging Face model identifier. |

---

## DiffusionPolicyParams

**Type key:** `diffusion_policy`

A diffusion-based policy for robotics action prediction. Combines a vision-language backbone for conditioning with a transformer denoiser.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vision_language_backbone` | `Union[VLMBackboneParams, CLIPBackboneParams, VLMFoundryBackboneParams, ViTBackboneParams]` | `CLIPBackboneParams()` | Visual and language conditioning backbone. |
| `transformer` | `Union[TransformerParams, TransformerHFParams]` | `ModelParams()` | Denoising transformer backbone. |
| `noise_scheduler` | `NoiseSchedulerParams` | `NoiseSchedulerParams()` | Noise schedule for the diffusion process. |
| `use_diffusers_scheduler` | `bool` | `False` | Use a HuggingFace Diffusers scheduler implementation. |
| `use_flow_matching_scheduler` | `bool` | `False` | Use flow matching instead of DDPM diffusion. |
| `input_noise_std` | `float` | `0.0` | Standard deviation of Gaussian noise added to inputs. |
| `diffusion_step_conditioning` | `Literal["add", "concat"]` | `"concat"` | How the diffusion timestep is injected into the transformer. |
| `action_dim` | `int` | `None` | **Shared.** Auto-set from `DataParams.action_dim`. |
| `proprioception_dim` | `int` | `0` | **Shared.** Auto-set from `DataParams.proprioception_dim`. |

---

## StableDiffusionParams

**Type key:** `stable_diffusion`

A Stable Diffusion model for image generation, with optional classifier-free guidance.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `unet` | `UNetParams` | `UNetParams()` | UNet architecture configuration. |
| `noise_scheduler` | `NoiseSchedulerParams` | `NoiseSchedulerParams()` | Noise schedule configuration. |
| `use_diffusers_unet` | `bool` | `False` | Use a HuggingFace Diffusers UNet. |
| `use_diffusers_scheduler` | `bool` | `False` | Use a HuggingFace Diffusers scheduler. |
| `use_flow_matching_scheduler` | `bool` | `False` | Use flow matching instead of DDPM. |
| `clip` | `CLIPHFParams` | `CLIPHFParams()` | CLIP text encoder configuration. |
| `do_classifier_free_guidance` | `bool` | `False` | Enable classifier-free guidance. |
| `guidance_scale` | `float` | `4.0` | CFG guidance scale. |
| `dropout_percent` | `float` | `0.2` | Conditioning dropout rate for unconditional training (CFG). |

---

## Supporting Parameter Classes

### UNetParams

**Type key:** `unet`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `in_channels` | `int` | `3` | Input channels. |
| `out_channels` | `int` | `3` | Output channels. |
| `time_emb_dim` | `int` | `256` | Timestep embedding dimension. |
| `text_emb_dim` | `int` | `512` | Text conditioning embedding dimension. |
| `channels` | `List[int]` | `[]` | Channel counts per UNet level (e.g., `[128, 256, 512, 1024]`). |
| `image_size` | `int` | `128` | Spatial resolution of the UNet input. |
| `time_mlp_float32` | `bool` | `False` | Run the time MLP in float32. |

### NoiseSchedulerParams

**Type key:** `noise_scheduler`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_timesteps` | `int` | `1000` | Total diffusion timesteps. |
| `beta_start` | `float` | `0.0001` | Starting noise level. |
| `beta_end` | `float` | `0.02` | Ending noise level. |
| `clamp_range` | `Tuple[float, float]` | `(-1.5, 1.5)` | Output clamping range. Interacts with normalization settings. |

### CLIPHFParams

**Type key:** `clip_hf`

Inherits from `TransformerHFParams`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hf_pretrained` | `str` | `None` | HF model identifier (e.g., `"openai/clip-vit-base-patch32"`). |
| `freeze_text_encoder` | `bool` | `False` | Freeze the text encoder weights. |
| `freeze_image_encoder` | `bool` | `False` | Freeze the image encoder weights. |

### CLIP_OpenCLIPParams

**Type key:** `clip_openclip`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `architecture` | `str` | `None` | OpenCLIP architecture name. |
| `pretrained_weights` | `str` | `None` | Pretrained weights identifier. |
| `freeze_text_encoder` | `bool` | `False` | Freeze the text encoder weights. |
| `freeze_image_encoder` | `bool` | `False` | Freeze the image encoder weights. |

### CLIPBackboneParams

**Type key:** `clip_backbone`

Inherits from both `BackboneParams` and `CLIPHFParams`. Used as the vision-language backbone in `DiffusionPolicyParams`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `disable_text` | `bool` | `False` | Disable the text branch (vision-only conditioning). |

### VLMBackboneParams

**Type key:** `vlm_backbone`

Inherits from both `BackboneParams` and `VLMHFParams`. Uses a pretrained VLM as a vision-language backbone for diffusion conditioning, extracting hidden states from the last N layers.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hf_pretrained` | `str` | `None` | HF model identifier. *(Inherited from VLMHFParams)* |
| `num_vlm_layers_to_use` | `int` | `4` | Number of last VLM layers to extract hidden states from for diffusion conditioning. |

### VLMFoundryBackboneParams

**Type key:** `vlm_foundry_backbone`

Inherits from `BackboneParams`. Uses a VLA Foundry-trained VLM as a vision-language backbone for diffusion conditioning, extracting hidden states from the last N layers.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_vlm_layers_to_use` | `int` | `4` | Number of last VLM layers to extract hidden states from for diffusion conditioning. |

### ViTBackboneParams

**Type key:** `vit_backbone`

Inherits from both `BackboneParams` and `ViTParams`. Uses a Vision Transformer as a vision-only backbone for diffusion conditioning. All fields are inherited from `ViTParams`.

