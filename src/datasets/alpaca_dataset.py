from datasets import load_dataset

from .base_dataset import BaseDataset


class AlpacaDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str = "tatsu-lab/alpaca",
        split: str = "train",
        offset: int = 0,
        limit: int = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.offset = offset
        self.limit = limit

        ds = load_dataset(dataset_name)
        data = ds[split]

        end_idx = len(data) if limit is None else min(offset + limit, len(data))
        self.data = data.select(range(offset, end_idx))

    def __getitem__(self, idx: int) -> dict:
        return {"text": self.data[idx]["text"]}

    def __len__(self) -> int:
        return len(self.data)
