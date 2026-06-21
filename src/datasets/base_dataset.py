from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
