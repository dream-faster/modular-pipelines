import pandas as pd
from datasets import Dataset


class RawDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        super(RawDataset, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset["input"].iloc(idx), self.dataset["label"].iloc(idx)[idx]

    def from_pandas(self, *args, **kwargs):
        return super(RawDataset, self).from_pandas(self, *args, **kwargs)
