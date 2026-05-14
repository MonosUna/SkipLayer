import torch.nn as nn


class ResidualMLPAligner(nn.Module):
    """Single shared MLP applied at every layer index >= start_layer
    with a residual connection.

    Unlike :class:`MLPAligner`, the output is ``x + mlp(x)`` rather than
    ``mlp(x)``. Empirically this makes the aligner easier to train
    because the MLP only needs to predict the increment $\\Delta_l$
    rather than reconstructing the entire hidden state.
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
        return x + self.mlp(x)
