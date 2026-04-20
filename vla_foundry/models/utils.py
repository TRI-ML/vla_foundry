from vla_foundry.params.model_params import ViTParams


def compute_num_image_tokens(vit_params: ViTParams) -> int:
    num_patches = (vit_params.img_size // vit_params.patch_size) ** 2
    scale = vit_params.projector_pixel_shuffle_factor
    return num_patches // (scale * scale)


def get_hidden_dim_hf(hf_config):
    if hasattr(hf_config, "hidden_size"):
        return hf_config.hidden_size
    if hasattr(hf_config, "text_config") and hasattr(hf_config.text_config, "hidden_size"):
        return hf_config.text_config.hidden_size
        # Fallback by probing last lm head weight if available
    if hasattr(hf_config, "lm_head") and hasattr(hf_config.lm_head, "weight"):
        return hf_config.lm_head.weight.shape[1]
    raise AttributeError("Could not infer hidden_dim from HF transformer config")


def get_num_hidden_layers_hf(hf_config):
    for key in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(hf_config, key):
            return getattr(hf_config, key)
    if hasattr(hf_config, "text_config"):
        for key in ["num_hidden_layers", "n_layer", "num_layers"]:
            if hasattr(hf_config.text_config, key):
                return getattr(hf_config.text_config, key)
    raise AttributeError("Could not infer num_hidden_layers from HF transformer config")


def get_hidden_states_hf(hf_output):
    hidden_states = None
    for attr in [
        "hidden_states",  # unified
        "text_hidden_states",  # some VLMs
        "decoder_hidden_states",  # encoder-decoder
        "encoder_hidden_states",  # encoder-only text
    ]:
        if hasattr(hf_output, attr) and getattr(hf_output, attr) is not None:
            hidden_states = getattr(hf_output, attr)
            break
    return hidden_states
