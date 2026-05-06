from collections.abc import Callable
from typing import Any

import torch
from src.metrics.base_metric import BaseMetric
from torch.utils.data import DataLoader
from tqdm import tqdm


class Loss(BaseMetric):
    """Loss metric computed via the provided loss function."""

    def __init__(self, loss_fn: Callable):
        self.loss_fn = loss_fn

    def __call__(
        self, model: Any, dataloader: DataLoader, max_batches: int = None
    ) -> dict[str, float]:
        # Train mode so the layer-skip iteration strategy actually substitutes
        # a random layer with the aligner (matching the training-time loss).
        model.train()

        total_loss = 0.0
        num_batches = 0
        device = next(model.parameters()).device

        with torch.no_grad(), self.autocast_context(model):
            for batch in tqdm(dataloader, desc="Computing loss", leave=False):
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                forward_kwargs = getattr(self.loss_fn, "model_forward_kwargs", {})
                if forward_kwargs:
                    batch = {**batch, **forward_kwargs}
                outputs = model.custom_forward(batch)
                if getattr(self.loss_fn, "requires_model", False):
                    loss = self.loss_fn(outputs, model=model)
                else:
                    loss = self.loss_fn(outputs)
                total_loss += loss.item()

                num_batches += 1
                if max_batches is not None and num_batches >= max_batches:
                    break

        return {"loss": total_loss / max(num_batches, 1)}
