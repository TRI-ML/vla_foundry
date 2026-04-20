import torch
import torch.nn as nn

from vla_foundry.model_utils import Float32Module


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.register_buffer("buf", torch.ones(1))

    def forward(self, x):
        return self.linear(x) + self.buf


def test_float32module_keeps_params_float32_and_preserves_class():
    mod = Dummy()
    mod_fp16 = Float32Module(mod)

    # Preserves original class
    assert mod_fp16.__class__.__name__ == "Dummy"

    # All params are float32
    assert all(p.dtype is torch.float32 for p in mod_fp16.parameters())


def test_float32module_to_moves_device_only():
    mod = Float32Module(Dummy())

    # Attempt to change dtype should be ignored; device move allowed
    mod = mod.to(dtype=torch.float16)  # ignored
    for p in mod.parameters():
        assert p.dtype is torch.float32

    # Move to CPU explicitly (no-op but should return self)
    mod2 = mod.to(device=torch.device("cpu"))
    assert mod2 is mod


@torch.no_grad()
def test_float32module_forward_casts_back():
    mod = Float32Module(Dummy(), cast_outputs_back=True)

    x_f16 = torch.randn(2, 4, dtype=torch.float16)
    out = mod(x_f16)

    # Output should match input dtype when cast-back enabled
    assert out.dtype is torch.float16

    # And computation should still be valid
    assert out.shape == (2, 4)


@torch.no_grad()
def test_float32module_prevents_bf16_underflow_to_zero():
    # Build a tiny linear layer with extremely small weights that become 0 when cast to bf16
    base = nn.Linear(1, 1, bias=False)
    base.weight.data.fill_(1e-45)  # below bf16 minimum subnormal, will cast to 0 in bf16

    # Unwrapped module cast to bf16: parameters quantize to bf16 first (weight becomes 0)
    unwrapped = nn.Linear(1, 1, bias=False)
    unwrapped.load_state_dict(base.state_dict())
    unwrapped = unwrapped.to(torch.bfloat16)

    # Wrapped module keeps params in fp32 and computes in fp32, then casts back
    wrapped = nn.Linear(1, 1, bias=False)
    wrapped.load_state_dict(base.state_dict())
    wrapped = Float32Module(wrapped, cast_outputs_back=True)
    wrapped = wrapped.to(torch.bfloat16)

    # Large magnitude input so fp32 compute produces a representable bf16 output
    x = torch.tensor([[1e30]], dtype=torch.bfloat16)

    out_unwrapped = unwrapped(x)
    out_wrapped = wrapped(x)

    # Without wrapper the tiny weight rounds/flushes to 0 in bf16, output is exactly 0
    assert torch.all(out_unwrapped == 0)

    # With wrapper the compute happens in fp32, output should be non-zero (then cast back to bf16)
    assert torch.any(out_wrapped != 0)
