import torch.nn as nn


class MLPAligner(nn.Module):
    """Single shared MLP applied at every layer index >= start_layer.

    Acts as a learnable substitute for a real decoder layer when the
    iteration strategy decides to skip the layer.
    """

    def __init__(self, hidden_size: int, hidden_mlp: int | None = None, start_layer: int = 0):
        super().__init__()
        hidden_mlp = hidden_mlp or hidden_size
        self.start_layer = start_layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_mlp),
            nn.GELU(),
            nn.Linear(hidden_mlp, hidden_size),
        )

    def forward(self, x, i):
        if i < self.start_layer:
            return x
        return self.mlp(x)
