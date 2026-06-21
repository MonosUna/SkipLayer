import torch.nn as nn


class BaseAligner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, i):
        return x
