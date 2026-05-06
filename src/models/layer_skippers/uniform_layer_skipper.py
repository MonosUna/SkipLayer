import torch
import torch.nn as nn


class UniformLayerSkipper(nn.Module):
    """Skip layer with probability ``p`` if the layer index lies inside one
    of the configured percentile ranges (e.g. ``[(0.5, 1.0)]`` skips only the
    second half of the network).
    """

    def __init__(self, num_layers: int, p: float = 0.5, skip_percentile_ranges=None):
        super().__init__()
        self.num_layers = num_layers
        self.p = p
        self.skip_percentile_ranges = skip_percentile_ranges or []

    def forward(self, x, i):
        return x

    def _in_skip_range(self, i: int) -> bool:
        pos = (i + 1) / self.num_layers
        for start, end in self.skip_percentile_ranges:
            if start < pos <= end:
                return True
        return False

    def should_skip(self, x, i):
        if not self._in_skip_range(i):
            return False
        return torch.rand(1).item() < self.p
