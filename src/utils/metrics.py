import hydra
from omegaconf import DictConfig
from src.metrics.base_metric import BaseMetric


def create_metrics(cfg: DictConfig) -> dict[str, list[BaseMetric]]:
    metrics = {}
    for split in cfg.metrics.keys():
        split_metrics = []
        for metric_cfg in cfg.metrics[split]:
            metric = hydra.utils.instantiate(metric_cfg)
            split_metrics.append(metric)
        metrics[split] = split_metrics
    return metrics
