import torch

from vla_foundry.models.model_outputs.base_output import BaseOutput


class VisionLanguageBackboneOutput(BaseOutput):
    """Unified output for vision-language backbones used in action policies.

    This provides a consistent interface regardless of backbone type (VLM or CLIP).
    """

    def __init__(self, embeddings: torch.Tensor):
        super().__init__()
        # Primary embeddings for conditioning [B, N, D]
        # VLM: [B, 1, hidden_dim * num_layers] - single embedding from action token
        # CLIP: [B, 1+N, projection_dim] - concatenated [text, images] sequence
        self.embeddings = embeddings
