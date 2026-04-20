from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=True):
        super().__init__()
        self.w12 = nn.Linear(in_dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x):
        gate, x = self.w12(x).chunk(2, dim=-1)
        x = F.silu(gate) * x
        return self.w3(x)


def get_feed_forward(ffn_type, hidden_dim):
    if ffn_type == "swiglu":
        # this follows llama / lit llama -- go to multiple of 256
        ffn_hidden_dim = 256 * ((int(2 * 4 * hidden_dim / 3) + 256 - 1) // 256)
        feed_forward = SwiGLU(hidden_dim, ffn_hidden_dim, hidden_dim, bias=False)
    elif ffn_type == "gelu":
        # Follows mosaic mpt7b, but without a bias.
        ffn_hidden_dim = hidden_dim * 4
        _ff_w1 = nn.Linear(hidden_dim, ffn_hidden_dim, bias=False)
        _ff_w2 = nn.Linear(ffn_hidden_dim, hidden_dim, bias=False)
        feed_forward = nn.Sequential(_ff_w1, nn.GELU(approximate="none"), _ff_w2)
    return feed_forward, ffn_hidden_dim
