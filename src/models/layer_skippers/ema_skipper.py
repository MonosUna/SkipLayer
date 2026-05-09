from collections import deque
from typing import Optional

import torch
import torch.nn as nn


class EmaSkipper(nn.Module):
    """Stateful skipper for the EMA-compensation layer-skip method.

    The method (see TODO.md) has the following moving parts, all bundled
    together in this single module so the iteration strategy stays simple:

    1. **Protected layers** — the first ``protected_first`` and last
       ``protected_last`` layers are never skipped.
    2. **Skip candidates** — chosen once after prefill: for each
       *eligible* layer ``l`` (i.e. not protected) we score
       ``I_l = ||Δ_l|| / ||h_l||`` from the prefill forward (Δ_l is the
       per-layer delta ``h_out - h_in``). The ``K`` lowest-scoring
       layers become the candidate set, where ``K = round(skip_ratio *
       num_eligible)``. The set is fixed for the rest of the
       generation.
    3. **EMA compensation** — for every executed layer ``l`` we maintain
       a running estimate ``Δ̂_l = β · Δ̂_l + (1 - β) · (h_out - h_in)``.
       When the iteration strategy decides to skip layer ``l`` it sets
       ``h_out = h_in + Δ̂_l``.
    4. **Adaptive skip probability** — a single global ``p_skip`` is
       maintained. After every generation step we compute the
       fraction of skipped candidates and average it over the last
       ``history_window`` steps. If that running average exceeds
       ``target_ratio`` we decrease ``p_skip`` by ``adaptation_rate``;
       otherwise we increase it. Clamped to ``[p_min, p_max]``.
    5. **Periodic refresh** — every ``refresh_interval`` generation
       steps no skipping is performed at all (full forward through
       every layer); this keeps EMA estimates from drifting.
    """

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

        # Mutable runtime state (one generation request at a time).
        self.candidates: set[int] = set()
        self.ema: dict[int, torch.Tensor] = {}
        self.p_skip: float = p_init
        self.step_counter: int = 0
        self.skip_history: deque[float] = deque(maxlen=self.history_window)

    # ------------------------------------------------------------------ utils

    def is_protected(self, i: int) -> bool:
        return i < self.protected_first or i >= self.num_layers - self.protected_last

    def num_eligible(self) -> int:
        return max(0, self.num_layers - self.protected_first - self.protected_last)

    def num_candidates(self) -> int:
        return round(self.skip_ratio * self.num_eligible())

    # --------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Reset all mutable state. Called at the start of every prefill so
        each new generation request starts from a clean slate."""
        self.candidates = set()
        self.ema = {}
        self.p_skip = self.p_init
        self.step_counter = 0
        self.skip_history = deque(maxlen=self.history_window)

    def record_prefill(
        self,
        layer_io: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Initialize candidates and EMA buffers from a prefill forward.

        ``layer_io[i]`` is ``(h_in_i, h_out_i)`` of the i-th decoder layer
        as returned during prefill. EMA is initialized to the per-layer
        delta on the **last** prefill position; importance ``I_l`` is
        averaged across the prefill sequence to be more robust.
        """
        importance: dict[int, float] = {}
        for i, (h_in, h_out) in enumerate(layer_io):
            delta = h_out - h_in
            # Last prefill position drives the EMA initial value (most
            # relevant for the very next generation step).
            self.ema[i] = delta[..., -1:, :].detach().clone()

            if self.is_protected(i):
                continue
            # Per-position L2 norm ratio, then average over batch & seq.
            d_norm = delta.float().norm(dim=-1)
            h_norm = h_in.float().norm(dim=-1).clamp_min(1e-6)
            importance[i] = (d_norm / h_norm).mean().item()

        k = self.num_candidates()
        if k == 0 or not importance:
            self.candidates = set()
        else:
            sorted_layers = sorted(importance.items(), key=lambda kv: kv[1])
            self.candidates = {layer for layer, _ in sorted_layers[:k]}

    # --------------------------------------------------------------- skipping

    def is_refresh_step(self) -> bool:
        return self.step_counter > 0 and self.step_counter % self.refresh_interval == 0

    def should_skip(self, i: int) -> bool:
        """Per-layer decision called by the iteration strategy on every
        generation step (single-token forward only)."""
        if self.is_protected(i):
            return False
        if i not in self.candidates:
            return False
        if self.is_refresh_step():
            return False
        return torch.rand(1).item() < self.p_skip

    def update_ema(self, i: int, h_in: torch.Tensor, h_out: torch.Tensor) -> None:
        """Update EMA estimate for an executed layer."""
        delta = (h_out - h_in).detach()
        if i not in self.ema:
            self.ema[i] = delta.clone()
        else:
            prev = self.ema[i]
            if prev.shape != delta.shape:
                # Prefill stored shape (..., 1, H); generation gives the same.
                prev = prev[..., -delta.shape[-2] :, :]
            self.ema[i] = self.beta * prev + (1.0 - self.beta) * delta

    def compensate(self, i: int, h_in: torch.Tensor) -> torch.Tensor:
        """Return ``h_in + Δ̂_i`` for a skipped layer."""
        delta = self.ema.get(i)
        if delta is None:
            return h_in
        if delta.shape != h_in.shape:
            delta = delta.expand_as(h_in)
        return h_in + delta.to(dtype=h_in.dtype, device=h_in.device)

    # --------------------------------------------------------------- adaption

    def end_step(self, num_skipped: int) -> None:
        """Called once per generation step after the layer loop. Updates
        the running skip-ratio average and adapts ``p_skip`` toward
        ``target_ratio``."""
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

    # --------------------------------------------------------- nn.Module glue

    def forward(self, x: torch.Tensor, i: Optional[int] = None) -> torch.Tensor:
        return x
