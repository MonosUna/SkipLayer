from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets."""

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Return a dict with at least the key ``"text"``."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
