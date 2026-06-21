import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def create_dataloaders(cfg: DictConfig) -> dict[str, DataLoader]:
    dataloaders = {}
    for split in cfg.dataset.keys():
        dataset = hydra.utils.instantiate(cfg.dataset[split])
        collator = hydra.utils.instantiate(cfg.collator[split])
        dataloader_fn = hydra.utils.instantiate(cfg.dataloader[split])
        dataloaders[split] = dataloader_fn(dataset=dataset, collate_fn=collator)
    return dataloaders
