import torch.nn as nn


class BaseLayerSkipper(nn.Module):
    """No-op skipper. ``should_skip`` always returns False at inference."""

    def __init__(self):
        super().__init__()

    def forward(self, x, i):
        return x

    def should_skip(self, x, i):
        return False
