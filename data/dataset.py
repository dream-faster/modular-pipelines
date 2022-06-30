import pandas as pd
from torch.utils.data import Dataset


class RawDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset['text'].iloc(idx), self.dataset['label'].iloc(idx)[idx]
