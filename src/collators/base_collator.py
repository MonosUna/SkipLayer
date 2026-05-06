from typing import Any

import torch


class BaseCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "max_length",
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.return_tensors = return_tensors

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors=self.return_tensors,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        # Causal LM shift: predict next token.
        return {
            "input_ids": input_ids[:, :-1],
            "attention_mask": attention_mask[:, :-1],
            "labels": input_ids[:, 1:],
        }
