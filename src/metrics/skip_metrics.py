from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch
from src.metrics.base_metric import BaseMetric
from src.models.layer_skippers.base_layer_skipper import BaseLayerSkipper
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.cache_utils import DynamicCache


def build_skip_html(skip_matrix, tokens, full_tokens=None, match_mask=None):
    _, num_layers = skip_matrix.shape

    html = """
    <style>
        table { border-collapse: collapse; font-family: monospace; }
        th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: center; }
        .skip { background-color: #ffb3b3; }
        .noskip { background-color: #b3ffb3; }
        .match { background-color: #b3ffb3; }
        .miss  { background-color: #ffb3b3; }
        .token { text-align: left; font-weight: bold; }
    </style>
    """

    has_ref = full_tokens is not None and match_mask is not None
    html += "<table>"
    html += "<tr><th>Skip token</th>"
    if has_ref:
        html += "<th>Full token</th><th>=?</th>"
    for layer_idx in range(num_layers):
        html += f"<th>L{layer_idx}</th>"
    html += "</tr>"

    for idx, (token, row) in enumerate(zip(tokens, skip_matrix, strict=False)):
        html += f"<tr><td class='token'>{token}</td>"
        if has_ref:
            full_tok = full_tokens[idx]
            match = bool(match_mask[idx])
            cls = "match" if match else "miss"
            html += f"<td class='token'>{full_tok}</td><td class='{cls}'>{'1' if match else '0'}</td>"
        for val in row:
            cls = "skip" if val.item() == 1 else "noskip"
            html += f"<td class='{cls}'>{int(val.item())}</td>"
        html += "</tr>"

    html += "</table>"
    return html


@contextmanager
def _disable_layer_skipping(model: Any):
    original_skipper = model.model.layer_skipper
    model.model.layer_skipper = BaseLayerSkipper().to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    try:
        yield
    finally:
        model.model.layer_skipper = original_skipper


class SkipMetrics(BaseMetric):
    def __init__(
        self,
        tokenizer: Any,
        max_new_tokens: int = 64,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

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

    @staticmethod
    def _greedy_step(
        model: Any,
        step_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: DynamicCache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        outputs = model.custom_forward(
            {
                "input_ids": step_input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
        )
        logits = outputs["logits"][:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        log_probs = torch.log_softmax(logits, dim=-1)
        skip_tensor = outputs.get("skip_tensor")
        if skip_tensor is not None and skip_tensor.dim() > 1:
            skip_tensor = skip_tensor.squeeze(0)
        return next_token, log_probs, skip_tensor

    def __call__(
        self,
        model: Any,
        dataloader: DataLoader,
        max_batches: int | None = None,
    ) -> dict[str, float | str]:
        model.eval()
        device = next(model.parameters()).device

        match_count_by_k: dict[int, int] = defaultdict(int)
        total_count_by_k: dict[int, int] = defaultdict(int)
        full_token_logprob_sum_by_k: dict[int, float] = defaultdict(float)
        full_token_prob_sum_by_k: dict[int, float] = defaultdict(float)

        total_match = 0
        total_count = 0
        total_full_token_prob = 0.0

        last_skip_html: str | None = None

        with torch.no_grad(), self.autocast_context(model):
            for batch_idx, batch in enumerate(tqdm(dataloader, leave=False)):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                for sample_idx in range(len(batch["messages"])):
                    input_ids, attention_mask = self._get_inputs_for_sample(
                        batch, sample_idx, device
                    )

                    skip_pkv = DynamicCache(config=model.config)
                    full_pkv = DynamicCache(config=model.config)

                    generated_ids = input_ids
                    skip_token_ids: list[int] = []
                    full_token_ids: list[int] = []
                    match_mask: list[bool] = []
                    skip_steps: list[torch.Tensor] = []

                    for step in range(self.max_new_tokens):
                        if step == 0:
                            step_input_ids = generated_ids
                        else:
                            step_input_ids = generated_ids[:, -1:]

                        skip_token, skip_log_probs, skip_tensor = self._greedy_step(
                            model, step_input_ids, attention_mask, skip_pkv
                        )

                        with _disable_layer_skipping(model):
                            full_token, _, _ = self._greedy_step(
                                model, step_input_ids, attention_mask, full_pkv
                            )

                        if skip_tensor is None:
                            skip_tensor = torch.zeros(
                                model.config.num_hidden_layers, device=device
                            )
                        k = int(skip_tensor.sum().item())

                        match = bool(skip_token.item() == full_token.item())
                        total_count_by_k[k] += 1
                        if match:
                            match_count_by_k[k] += 1
                        total_count += 1
                        total_match += int(match)

                        full_token_logprob = float(
                            skip_log_probs.gather(dim=-1, index=full_token).item()
                        )
                        full_token_prob = float(torch.exp(torch.tensor(full_token_logprob, dtype=torch.float32)).item())
                        full_token_logprob_sum_by_k[k] += full_token_logprob
                        full_token_prob_sum_by_k[k] += full_token_prob
                        total_full_token_prob += full_token_prob

                        skip_token_ids.append(skip_token.item())
                        full_token_ids.append(full_token.item())
                        match_mask.append(match)
                        skip_steps.append(skip_tensor.detach().cpu())

                        generated_ids = torch.cat([generated_ids, full_token], dim=1)
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones_like(full_token)], dim=1
                        )

                    skip_matrix = torch.stack(skip_steps, dim=0)
                    skip_token_strings = [
                        self.tokenizer.decode([tid], skip_special_tokens=False)
                        for tid in skip_token_ids
                    ]
                    full_token_strings = [
                        self.tokenizer.decode([tid], skip_special_tokens=False)
                        for tid in full_token_ids
                    ]
                    last_skip_html = build_skip_html(
                        skip_matrix, skip_token_strings, full_token_strings, match_mask
                    )

        num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
        width = len(str(num_layers)) if num_layers is not None else 2

        result: dict[str, float | str] = {}
        for k, total in total_count_by_k.items():
            if total <= 0:
                continue
            result[f"accuracy/skip_{k:0{width}d}"] = match_count_by_k[k] / total
            result[f"count/skip_{k:0{width}d}"] = float(total)
            result[f"avg_full_token_logprob/skip_{k:0{width}d}"] = (
                full_token_logprob_sum_by_k[k] / total
            )
            result[f"avg_full_token_prob/skip_{k:0{width}d}"] = (
                full_token_prob_sum_by_k[k] / total
            )

        if total_count > 0:
            result["accuracy/overall"] = total_match / total_count
            result["count/overall"] = float(total_count)
            result["avg_full_token_prob/overall"] = total_full_token_prob / total_count

        if last_skip_html is not None:
            result["html/skip_table"] = last_skip_html
        return result
