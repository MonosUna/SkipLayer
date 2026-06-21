from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader


class BaseMetric(ABC):
    @staticmethod
    def autocast_context(model: Any):
        device = next(model.parameters()).device
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @abstractmethod
    def __call__(self, model: Any, dataloader: DataLoader) -> dict[str, float]:
        pass
