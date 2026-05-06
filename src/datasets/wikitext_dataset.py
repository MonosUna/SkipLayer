from datasets import load_dataset

from .base_dataset import BaseDataset


class WikitextDataset(BaseDataset):
    """Dataset for Salesforce/wikitext language modeling data."""

    def __init__(
        self,
        dataset_name: str = "Salesforce/wikitext",
        config_name: str = "wikitext-2-raw-v1",
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
        data = data.filter(lambda x: len(x["text"].strip()) > 50)

        end_idx = len(data) if limit is None else min(offset + limit, len(data))
        self.data = data.select(range(offset, end_idx))

    def __getitem__(self, idx: int) -> dict:
        return {"text": self.data[idx]["text"]}

    def __len__(self) -> int:
        return len(self.data)
