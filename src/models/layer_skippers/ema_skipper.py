from collections import deque
from typing import Optional

import torch
import torch.nn as nn


class EmaSkipper(nn.Module):
    def __init__(
        self,
        num_layers: int,
        protected_first: int = 3,
        protected_last: int = 2,
        skip_ratio: float = 0.3,
        beta: float = 0.9,
        refresh_interval: int = 8,
        p_init: float = 0.5,
        p_min: float = 0.1,
        p_max: float = 0.9,
        target_ratio: float = 0.4,
        adaptation_rate: float = 0.03,
        history_window: int = 16,
    ):
        super().__init__()
        if protected_first + protected_last >= num_layers:
            raise ValueError(
                "protected_first + protected_last must be < num_layers; "
                f"got {protected_first} + {protected_last} vs {num_layers}"
            )
        self.num_layers = num_layers
        self.protected_first = protected_first
        self.protected_last = protected_last
        self.skip_ratio = skip_ratio
        self.beta = beta
        self.refresh_interval = max(1, int(refresh_interval))
        self.p_init = p_init
        self.p_min = p_min
        self.p_max = p_max
        self.target_ratio = target_ratio
        self.adaptation_rate = adaptation_rate
        self.history_window = max(1, int(history_window))

        self.candidates: set[int] = set()
        self.ema: dict[int, torch.Tensor] = {}
        self.p_skip: float = p_init
        self.step_counter: int = 0
        self.skip_history: deque[float] = deque(maxlen=self.history_window)
        self.last_importance: dict[int, float] = {}

    def is_protected(self, i: int) -> bool:
        return i < self.protected_first or i >= self.num_layers - self.protected_last

    def num_eligible(self) -> int:
        return max(0, self.num_layers - self.protected_first - self.protected_last)

    def num_candidates(self) -> int:
        return round(self.skip_ratio * self.num_eligible())

    def reset(self) -> None:
        self.candidates = set()
        self.ema = {}
        self.p_skip = self.p_init
        self.step_counter = 0
        self.skip_history = deque(maxlen=self.history_window)
        self.last_importance = {}

    def record_prefill(
        self,
        layer_io: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        importance: dict[int, float] = {}
        for i, (h_in, h_out) in enumerate(layer_io):
            delta = h_out - h_in
            self.ema[i] = delta[..., -1:, :].detach().clone()

            d_norm = delta.float().norm(dim=-1)
            h_norm = h_in.float().norm(dim=-1).clamp_min(1e-6)
            importance[i] = (d_norm / h_norm).mean().item()

        self.last_importance = dict(importance)

        eligible_importance = {
            i: v for i, v in importance.items() if not self.is_protected(i)
        }
        k = self.num_candidates()
        if k == 0 or not eligible_importance:
            self.candidates = set()
        else:
            sorted_layers = sorted(eligible_importance.items(), key=lambda kv: kv[1])
            self.candidates = {layer for layer, _ in sorted_layers[:k]}

    def is_refresh_step(self) -> bool:
        return self.step_counter > 0 and self.step_counter % self.refresh_interval == 0

    def should_skip(self, i: int) -> bool:
        if self.is_protected(i):
            return False
        if i not in self.candidates:
            return False
        if self.is_refresh_step():
            return False
        return torch.rand(1).item() < self.p_skip

    def update_ema(self, i: int, h_in: torch.Tensor, h_out: torch.Tensor) -> None:
        delta = (h_out - h_in).detach()
        if i not in self.ema:
            self.ema[i] = delta.clone()
        else:
            prev = self.ema[i]
            if prev.shape != delta.shape:
                prev = prev[..., -delta.shape[-2] :, :]
            self.ema[i] = self.beta * prev + (1.0 - self.beta) * delta

    def compensate(self, i: int, h_in: torch.Tensor) -> torch.Tensor:
        delta = self.ema.get(i)
        if delta is None:
            return h_in
        if delta.shape != h_in.shape:
            delta = delta.expand_as(h_in)
        return h_in + delta.to(dtype=h_in.dtype, device=h_in.device)

    def end_step(self, num_skipped: int) -> None:
        denom = max(1, self.num_candidates())
        self.skip_history.append(num_skipped / denom)
        if self.skip_history:
            recent = sum(self.skip_history) / len(self.skip_history)
            if recent > self.target_ratio:
                self.p_skip -= self.adaptation_rate
            elif recent < self.target_ratio:
                self.p_skip += self.adaptation_rate
            self.p_skip = float(min(self.p_max, max(self.p_min, self.p_skip)))
        self.step_counter += 1

    def forward(self, x: torch.Tensor, i: Optional[int] = None) -> torch.Tensor:
        return x
