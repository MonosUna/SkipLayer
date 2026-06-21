import torch.nn as nn


class ResidualPerLayerMLPAligner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        hidden_mlp: int | None = None,
        start_layer: int = 0,
    ):
        super().__init__()
        hidden_mlp = hidden_mlp or hidden_size
        self.start_layer = start_layer
        self.num_layers = num_layers
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_mlp),
                    nn.GELU(),
                    nn.Linear(hidden_mlp, hidden_size),
                )
                for _ in range(num_layers - start_layer)
            ]
        )

    def forward(self, x, i):
        if i < self.start_layer:
            return x
        return x + self.mlps[i - self.start_layer](x)
