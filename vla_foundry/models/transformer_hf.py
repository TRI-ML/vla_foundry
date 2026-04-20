import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from vla_foundry.models.registry import register_model
from vla_foundry.models.transformer_base import TransformerBase
from vla_foundry.models.utils import get_hidden_dim_hf, get_num_hidden_layers_hf
from vla_foundry.params.model_params import TransformerHFParams


class TransformerHF(TransformerBase):
    def __init__(self, model_params: TransformerHFParams, load_pretrained: bool = True):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        if load_pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        else:
            config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_config(config)

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return out

    def embeddings(self, input_ids):
        """Get token embeddings for input_ids."""
        return self.model.get_input_embeddings()(input_ids)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError

    @property
    def hidden_dim(self) -> int:
        return get_hidden_dim_hf(self.model.config)

    @property
    def num_hidden_layers(self) -> int:
        return get_num_hidden_layers_hf(self.model.config)

    def resize_token_embeddings(self, token_id: int = None) -> int:
        """Extend the embedding vocabulary of the underlying LM."""
        embed = self.model.resize_token_embeddings(token_id)
        if token_id is None:
            return embed.num_embeddings
        else:
            return token_id

    def get_fsdp_block_types(self):
        """Return block types for FSDP wrapping."""
        for _name, module in self.model.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                return (type(module[0]),)
        raise ValueError("Could not find model block class.")


@register_model("transformer_hf")
def create_transformer_hf(model_params: TransformerHFParams, load_pretrained: bool = True):
    return TransformerHF(model_params, load_pretrained=load_pretrained)
