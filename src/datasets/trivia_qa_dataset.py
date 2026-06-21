from datasets import load_dataset

from .base_dataset import BaseDataset


class TriviaQADataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str = "mandarjoshi/trivia_qa",
        config_name: str = "rc",
        split: str = "train",
        offset: int = 0,
        limit: int = None,
    ):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.offset = offset
        self.limit = limit

        ds = load_dataset(dataset_name, config_name)
        data = ds[split]

        end_idx = len(data) if limit is None else min(offset + limit, len(data))
        self.data = data.select(range(offset, end_idx))

    def __getitem__(self, idx: int) -> dict:
        return {"text": self.data[idx]["question"]}

    def __len__(self) -> int:
        return len(self.data)
