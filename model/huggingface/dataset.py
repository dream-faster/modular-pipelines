import pandas as pd
from datasets import Dataset
from configs.constants import DataConst


class RawDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        super(RawDataset, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            self.dataset[DataConst.input_name].iloc(idx),
            self.dataset[DataConst.label_name].iloc(idx)[idx],
        )

    def from_pandas(self, *args, **kwargs):
        return super(RawDataset, self).from_pandas(self, *args, **kwargs)
