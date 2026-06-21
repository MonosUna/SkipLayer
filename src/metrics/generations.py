import time
from typing import Any

import torch
from src.metrics.base_metric import BaseMetric
from torch.utils.data import DataLoader
from tqdm import tqdm


class Generations(BaseMetric):
    def __init__(
        self,
        tokenizer: Any,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

    def _get_inputs_for_sample(
        self,
        batch: dict[str, Any],
        index: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        messages = batch["messages"][index]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
        )
        prompt_ids = prompt_ids.to(device)
        attn = torch.ones_like(prompt_ids, device=device, dtype=torch.long)
        return prompt_ids, attn

    def __call__(
        self,
        model: Any,
        dataloader: DataLoader,
        max_batches: int | None = None,
    ) -> dict[str, float | list[str]]:
        model.eval()
        device = next(model.parameters()).device

        texts: list[Any] = []
        total_gen_time = 0.0
        total_new_tokens = 0

        with torch.no_grad(), self.autocast_context(model):
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generations", leave=False)):
                batch_size = len(batch["messages"])
                for i in range(batch_size):
                    input_ids, attention_mask = self._get_inputs_for_sample(batch, i, device)
                    prompt_len = input_ids.shape[1]

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    total_gen_time += time.perf_counter() - t0
                    new_ids = generated[:, prompt_len:]
                    num_new = new_ids.shape[1]
                    total_new_tokens += num_new

                    prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    new_text = self.tokenizer.decode(new_ids[0], skip_special_tokens=True)
                    text = (
                        f"Prompt ({prompt_len} tokens):\n{prompt_text}\n"
                        f"Generated (+{num_new} tokens):\n{new_text}"
                    )
                    texts.append(
                        {
                            "text": text,
                            "metadata": {
                                "batch_idx": batch_idx + 1,
                                "sample_idx": i + 1,
                            },
                        }
                    )

                if max_batches is not None and batch_idx + 1 >= max_batches:
                    break

        time_per_token_ms = (
            (total_gen_time * 1000.0 / total_new_tokens) if total_new_tokens else 0.0
        )
        return {
            "generations": texts,
            "generation_time_per_token_ms": time_per_token_ms,
        }
