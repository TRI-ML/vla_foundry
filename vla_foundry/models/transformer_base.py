from vla_foundry.models.base_model import BaseModel


class TransformerBase(BaseModel):
    """Abstract interface for Vision-Language Models.

    Implementations must expose:
    - hidden_dim: language model hidden size
    - num_hidden_layers: number of language model layers whose hidden states can be returned
    - forward(..., return_hidden_states=False) -> (logits, hidden_states|None)
      where hidden_states is a list/tuple of tensors shaped [batch, seq_len, hidden_dim]
      ordered from bottom to top layers.
    """

    @property
    def hidden_dim(self) -> int:
        raise NotImplementedError

    @property
    def num_hidden_layers(self) -> int:
        raise NotImplementedError

    def resize_token_embeddings(self, token_id: int = None) -> int:
        """Optional: add a token to the underlying LM if supported."""
        return token_id

    def forward(self, input_ids, image, attention_mask=None, return_hidden_states: bool = False):
        raise NotImplementedError
