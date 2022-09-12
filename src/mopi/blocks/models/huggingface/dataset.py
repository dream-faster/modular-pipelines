import pandas as pd
from datasets import Dataset

from mopi.constants import Const


class RawDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        super(RawDataset, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            self.dataset[Const.input_col].iloc(idx),
            self.dataset[Const.label_col].iloc(idx)[idx],
        )

    def from_pandas(self, *args, **kwargs):
        return super(RawDataset, self).from_pandas(self, *args, **kwargs)
