import os
import warnings
from collections import deque
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from src.metrics.base_metric import BaseMetric
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer:
    TRAIN_METRICS_MAX_BATCHES = 5000

    def __init__(
        self,
        cfg: DictConfig,
        model: Any,
        dataloaders: dict[str, DataLoader],
        metrics: dict[str, list[BaseMetric]],
        optimizer: Any,
        scheduler: Any | None = None,
        loss_fn: Any | None = None,
        logger: Any | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.logger = logger

        self.current_epoch = 0
        self.global_step = 0
        self.device = cfg.device
        loss_logging_window = int(
            getattr(self.cfg.training, "loss_logging_window", self.cfg.training.loss_logging_steps)
        )
        self.recent_train_losses = deque(maxlen=max(1, loss_logging_window))

        self.mixed_precision = self.cfg.training.mixed_precision
        if self.mixed_precision:
            mixed_precision_dtype_cfg = self.cfg.training.mixed_precision_dtype
            self.mixed_precision_dtype = hydra.utils.get_object(mixed_precision_dtype_cfg)

    def _compute_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        if getattr(self.loss_fn, "requires_model", False):
            return self.loss_fn(outputs, model=self.model)
        return self.loss_fn(outputs)

    def eval(self) -> None:
        self.evaluate_all_splits()
        if self.logger:
            self.logger.finish()

    def train(self) -> None:
        num_epochs = self.cfg.training.num_epochs
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self.train_epoch()
            self.log_metrics({"epoch": epoch + 1}, step=self.global_step)
            self.evaluate_all_splits(max_batches=self.cfg.training.eval_steps_max_batches)

            if (epoch + 1) % self.cfg.training.checkpoint_save_epochs == 0:
                checkpoint_filename = self.get_checkpoint_filename(epoch + 1)
                checkpoint_path = os.path.join(self.cfg.checkpoint_dir, checkpoint_filename)
                self.save_checkpoint(checkpoint_path)

        if self.logger:
            self.logger.finish()

    def train_epoch(self) -> None:
        self.model.train()
        dataloader = self.dataloaders["train"]
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            loss = self.train_step(batch)
            self.recent_train_losses.append(loss)
            pbar.set_postfix({"loss": loss})

            if self.global_step % self.cfg.training.loss_logging_steps == 0:
                mean_loss = sum(self.recent_train_losses) / len(self.recent_train_losses)
                self.log_metrics({"train/stepwise_loss": mean_loss}, step=self.global_step)

            if self.global_step % self.cfg.training.eval_steps == 0:
                self.evaluate_all_splits(max_batches=self.cfg.training.eval_steps_max_batches)
                self.model.train()

    def train_step(self, batch: dict[str, Any]) -> float:
        self.optimizer.zero_grad(set_to_none=True)

        batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
        forward_kwargs = getattr(self.loss_fn, "model_forward_kwargs", {})
        if forward_kwargs:
            batch = {**batch, **forward_kwargs}

        if self.mixed_precision:
            with torch.autocast(device_type=self.device, dtype=self.mixed_precision_dtype):
                outputs = self.model.custom_forward(batch)
                loss = self._compute_loss(outputs)
        else:
            outputs = self.model.custom_forward(batch)
            loss = self._compute_loss(outputs)

        if not loss.requires_grad:
            warnings.warn(
                "Skipping train step: loss does not require grad "
                "(no trainable parameter participated in the forward pass).",
                stacklevel=2,
            )
            self.global_step += 1
            return float(loss.item())

        loss.backward()

        if self.cfg.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1
        return loss.item()

    def evaluate_all_splits(self, max_batches: int | None = None) -> None:
        for split in self.dataloaders.keys():
            split_metrics = self.evaluate(
                split, max_batches=self._resolve_max_batches(split, max_batches)
            )
            self.log_metrics(split_metrics, step=self.global_step)

    def evaluate(self, split: str, max_batches: int | None = None) -> dict[str, float]:
        dataloader = self.dataloaders[split]
        split_metrics = self.metrics[split]
        results: dict[str, Any] = {}
        for metric in split_metrics:
            metric_results = metric(self.model, dataloader, max_batches=max_batches)
            for key, value in metric_results.items():
                results[f"{split}/{key}"] = value
        return results

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        full_state_dict = self.model.state_dict()
        trainable_param_names = {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }
        trainable_state_dict = {
            name: tensor
            for name, tensor in full_state_dict.items()
            if name in trainable_param_names
        }

        checkpoint_weights_only = bool(getattr(self.cfg, "checkpoint_weights_only", True))
        if checkpoint_weights_only:
            checkpoint = trainable_state_dict
        else:
            checkpoint = {
                "epoch": self.current_epoch,
                "next_epoch": self.current_epoch + 1,
                "global_step": self.global_step,
                "model_state_dict": trainable_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        self.upload_checkpoint_to_hub(path)

        if getattr(self.cfg, "log_checkpoint", False) and self.logger is not None:
            model_name = os.path.basename(path)
            self.logger.log_model(model_path=path, model_name=model_name)

    def get_checkpoint_filename(self, epoch: int) -> str:
        run_name = getattr(getattr(self.cfg, "logger", None), "run_name", None)
        if not run_name:
            return f"checkpoint_epoch_{epoch}.pt"
        sanitized = "".join(
            c if c.isalnum() or c in {"-", "_", "."} else "_" for c in run_name
        ).strip("._")
        if not sanitized:
            return f"checkpoint_epoch_{epoch}.pt"
        return f"{sanitized}_epoch_{epoch}.pt"

    def upload_checkpoint_to_hub(self, path: str) -> None:
        hf_cfg = getattr(self.cfg, "huggingface", None)
        if hf_cfg is None:
            return
        token = getattr(hf_cfg, "token", None)
        if not token:
            return
        repo_id = getattr(hf_cfg, "repo_id", None)
        if not repo_id:
            warnings.warn(
                "Skipping checkpoint upload to Hugging Face Hub because "
                "`huggingface.repo_id` is not set.",
                stacklevel=2,
            )
            return

        from huggingface_hub import create_repo, upload_file

        try:
            create_repo(
                repo_id=repo_id,
                token=token,
                exist_ok=True,
                private=bool(getattr(hf_cfg, "private", False)),
            )
            upload_file(
                path_or_fileobj=path,
                path_in_repo=os.path.basename(path),
                repo_id=repo_id,
                token=token,
                commit_message=f"Upload {os.path.basename(path)}",
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to upload checkpoint to Hugging Face Hub: {exc}",
                stacklevel=2,
            )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        is_full_checkpoint = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        state_dict = checkpoint["model_state_dict"] if is_full_checkpoint else checkpoint

        if is_full_checkpoint:
            self.current_epoch = checkpoint.get("next_epoch", checkpoint["epoch"] + 1)
            self.global_step = checkpoint["global_step"]

        load_result = self.model.load_state_dict(state_dict, strict=False)
        if load_result.unexpected_keys:
            raise ValueError(f"Unexpected checkpoint keys: {sorted(load_result.unexpected_keys)}")

        trainable_param_names = {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }
        missing_trainable_keys = sorted(
            key for key in load_result.missing_keys if key in trainable_param_names
        )
        if missing_trainable_keys:
            raise ValueError(
                f"Checkpoint is missing trainable parameters: {missing_trainable_keys}"
            )

        if not is_full_checkpoint:
            return

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
    ) -> None:
        if self.logger is not None:
            self.logger.log_metrics(metrics, step=step)

    def _resolve_max_batches(self, split: str, max_batches: int | None) -> int | None:
        if split != "train":
            return max_batches
        if max_batches is None:
            return self.TRAIN_METRICS_MAX_BATCHES
        return min(max_batches, self.TRAIN_METRICS_MAX_BATCHES)
