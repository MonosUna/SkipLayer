from collections import defaultdict
from typing import Any

import torch
from src.metrics.base_metric import BaseMetric
from torch.utils.data import DataLoader
from tqdm import tqdm


def _build_per_layer_table_html(
    num_layers: int,
    candidate_counts: list[int],
    importance_sum: list[float],
    importance_count: list[int],
    protected_first: int,
    protected_last: int,
    samples_seen: int,
) -> str:
    html = """
    <style>
        table { border-collapse: collapse; font-family: monospace; }
        th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }
        th { background-color: #eee; }
        td.protected { background-color: #f0f0f0; color: #888; }
        td.layer { font-weight: bold; text-align: left; }
        td.candidate { background-color: #ffe0b3; }
    </style>
    """
    html += "<table>"
    html += (
        "<tr>"
        "<th>layer</th>"
        "<th>kind</th>"
        "<th>cand. count</th>"
        "<th>cand. share</th>"
        "<th>avg importance I_l</th>"
        "</tr>"
    )
    for i in range(num_layers):
        if i < protected_first:
            kind = "protected (top)"
        elif i >= num_layers - protected_last:
            kind = "protected (bot)"
        else:
            kind = "eligible"
        cnt = candidate_counts[i]
        share = (cnt / samples_seen) if samples_seen > 0 else 0.0
        avg_imp = (
            importance_sum[i] / importance_count[i] if importance_count[i] > 0 else 0.0
        )
        cls = (
            "protected"
            if kind.startswith("protected")
            else ("candidate" if cnt > 0 else "")
        )
        html += (
            f"<tr><td class='layer'>L{i}</td>"
            f"<td class='{cls}'>{kind}</td>"
            f"<td class='{cls}'>{cnt}</td>"
            f"<td class='{cls}'>{share:.3f}</td>"
            f"<td class='{cls}'>{avg_imp:.4f}</td></tr>"
        )
    html += "</table>"
    return html


class EmaCandidateStats(BaseMetric):
    """Per-layer statistics produced by ``EmaSkipper`` after prefill.

    For every sample in the generation dataloader we run a single
    prefill forward (no token sampling) so the skipper's
    ``record_prefill`` is invoked. After the forward we read:

      - ``layer_skipper.candidates`` — set of layers chosen as skip
        candidates for this sample;
      - ``layer_skipper.last_importance`` — mapping ``layer_idx ->
        I_l = avg ||Δ_l|| / ||h_l||`` (computed for *all* layers,
        including protected ones, so we can inspect the full delta
        profile of the network).

    Reported metrics:

      - ``candidate_count/L{ii}`` — how many samples picked layer ``ii``
        as a skip candidate;
      - ``candidate_share/L{ii}`` — same divided by the number of
        samples;
      - ``avg_importance/L{ii}`` — mean importance across samples;
      - ``html/per_layer_table`` — combined table with columns
        layer / kind / candidate count / share / average importance
        (last sample's view).

    The metric needs the chat template applied to the prompt to drive
    a representative prefill, exactly like
    [`SkipMetrics`](src/metrics/skip_metrics.py:71).
    """

    def __init__(self, tokenizer: Any, **kwargs):
        self.tokenizer = tokenizer

    def _get_inputs_for_sample(
        self,
        batch: dict[str, Any],
        index: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        messages = batch["messages"][index]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
        ).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return input_ids, attention_mask

    def __call__(
        self,
        model: Any,
        dataloader: DataLoader,
        max_batches: int | None = None,
    ) -> dict[str, float | str]:
        skipper = getattr(model.model, "layer_skipper", None)
        if skipper is None or not hasattr(skipper, "record_prefill"):
            return {}

        model.eval()
        device = next(model.parameters()).device
        num_layers = model.config.num_hidden_layers

        candidate_counts = [0] * num_layers
        importance_sum = [0.0] * num_layers
        importance_count = [0] * num_layers
        samples_seen = 0

        protected_first = getattr(skipper, "protected_first", 0)
        protected_last = getattr(skipper, "protected_last", 0)

        with torch.no_grad(), self.autocast_context(model):
            for batch_idx, batch in enumerate(tqdm(dataloader, leave=False)):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                for sample_idx in range(len(batch["messages"])):
                    input_ids, attention_mask = self._get_inputs_for_sample(
                        batch, sample_idx, device
                    )
                    # Run a single prefill (multi-token) forward. The
                    # iteration strategy will reset the skipper and call
                    # ``record_prefill`` so the candidate set and
                    # importance scores are populated.
                    model.custom_forward(
                        {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "past_key_values": None,
                            "use_cache": True,
                            "compute_logits": False,
                        }
                    )

                    samples_seen += 1
                    for layer_idx in skipper.candidates:
                        if 0 <= layer_idx < num_layers:
                            candidate_counts[layer_idx] += 1
                    for layer_idx, value in skipper.last_importance.items():
                        if 0 <= layer_idx < num_layers:
                            importance_sum[layer_idx] += float(value)
                            importance_count[layer_idx] += 1

        result: dict[str, float | str] = {}
        if samples_seen == 0:
            return result

        width = len(str(num_layers))
        for i in range(num_layers):
            result[f"candidate_count/L{i:0{width}d}"] = float(candidate_counts[i])
            result[f"candidate_share/L{i:0{width}d}"] = (
                candidate_counts[i] / samples_seen
            )
            if importance_count[i] > 0:
                result[f"avg_importance/L{i:0{width}d}"] = (
                    importance_sum[i] / importance_count[i]
                )

        result["count/samples"] = float(samples_seen)
        # Overall protected/eligible aggregates for quick sanity checks.
        eligible_picks = sum(
            candidate_counts[i]
            for i in range(num_layers)
            if not (i < protected_first or i >= num_layers - protected_last)
        )
        protected_picks = sum(candidate_counts) - eligible_picks
        result["candidate_count/eligible_total"] = float(eligible_picks)
        result["candidate_count/protected_total"] = float(protected_picks)

        result["html/per_layer_table"] = _build_per_layer_table_html(
            num_layers,
            candidate_counts,
            importance_sum,
            importance_count,
            protected_first,
            protected_last,
            samples_seen,
        )

        # Reset state so subsequent metrics start clean.
        if hasattr(skipper, "reset"):
            skipper.reset()
        return result
