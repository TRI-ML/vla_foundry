"""
Marker base class for FSDP block wrapping.

Any module that should be independently wrapped by FSDP should inherit from FSDPBlock.
This eliminates the need for manual block type registration.
"""

import torch.nn as nn


class FSDPBlock(nn.Module):
    """
    Marker base class for modules that should be independently wrapped by FSDP.

    Usage:
        class TransformerBlock(FSDPBlock):
            def __init__(self, ...):
                super().__init__()
                # ... your implementation

    FSDP will automatically wrap all instances of FSDPBlock subclasses.
    """

    pass
