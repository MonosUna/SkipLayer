#!/usr/bin/env python3
# Comet: import before torch so auto-logging works (weights, etc.)
import comet_ml  # noqa: F401
import hydra
from omegaconf import DictConfig, OmegaConf
from src.trainers.base_trainer import BaseTrainer
from src.utils import (
    create_dataloaders,
    create_metrics,
    create_model,
    get_parameter_statistics,
    inject_model_config,
    iter_trainable_parameters,
    set_seed,
)


@hydra.main(version_base=None, config_path="../../configs", config_name="layer_skip")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.training.seed, cfg.device)
    inject_model_config(cfg)

    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_params(OmegaConf.to_container(cfg, resolve=True))

    dataloaders = create_dataloaders(cfg)
    print("\nDataLoaders created:")
    for split, dl in dataloaders.items():
        print(f"  - {split}: {len(dl)} batches")

    loss_fn = hydra.utils.instantiate(cfg.loss)
    print(f"\nLoss function: {loss_fn.__class__.__name__}")

    model = create_model(cfg)
    print(
        f"Model: {model.__class__.__name__} "
        f"(layers={model.config.num_hidden_layers}, "
        f"hidden={model.config.hidden_size}, device={cfg.device})"
    )
    parameter_stats = get_parameter_statistics(model)
    print("\nParameters:")
    print(
        f"  - total: {parameter_stats['total_params']:,} "
        f"({parameter_stats['total_gib']:.2f} GiB)"
    )
    print(
        f"  - trainable: {parameter_stats['trainable_params']:,} "
        f"({parameter_stats['trainable_gib']:.2f} GiB)"
    )
    if cfg.training_mode and parameter_stats["trainable_params"] == 0:
        raise ValueError("No trainable parameters found after model initialization.")

    metrics = create_metrics(cfg)
    print("\nMetrics:")
    for split, split_metrics in metrics.items():
        print(f"  - {split}: {[m.__class__.__name__ for m in split_metrics]}")

    optimizer = None
    scheduler = None
    if cfg.training_mode:
        optimizer = hydra.utils.instantiate(cfg.optimizer)(
            params=iter_trainable_parameters(model)
        )
        print(f"Optimizer: {optimizer.__class__.__name__} (lr={cfg.training.learning_rate})")
        scheduler = hydra.utils.instantiate(cfg.scheduler)(optimizer=optimizer)
        print(
            f"Scheduler: {scheduler.__class__.__name__} "
            f"(warmup={cfg.training.warmup_steps} steps)"
        )
    else:
        print("training_mode=False: skipping optimizer/scheduler initialization (eval-only run).")

    trainer = BaseTrainer(
        cfg=cfg,
        model=model,
        dataloaders=dataloaders,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        logger=logger,
    )

    if cfg.init_checkpoint_path:
        trainer.load_checkpoint(cfg.init_checkpoint_path)
        print(f"Loaded checkpoint: {cfg.init_checkpoint_path}")

    if cfg.training_mode:
        trainer.train()
    else:
        trainer.eval()


if __name__ == "__main__":
    main()
