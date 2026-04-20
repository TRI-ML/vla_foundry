import torch

from vla_foundry.models.model_outputs.base_output import BaseOutput


class TransformerOutput(BaseOutput):
    """
    Custom output class that mimics transformers.modeling_outputs.CausalLMOutputWithPast
    but doesn't depend on HuggingFace transformers library.

    This allows us to provide HF-compatible API without requiring HF dependencies
    for non-HF models.
    """

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        hidden_states: tuple[torch.Tensor] | list | None = None,
        attentions: tuple[torch.Tensor] | None = None,
        loss: torch.Tensor | None = None,
    ):
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.loss = loss
