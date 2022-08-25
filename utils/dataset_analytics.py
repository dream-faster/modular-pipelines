from blocks.base import DataLoader
from type import DatasetSplit
from constants import Const

def print_dataset_analytics(dataloader: DataLoader, split: DatasetSplit) -> None:
    data = dataloader.load(split)
    labels = data[Const.label_col]
    print(f"Dataloader: {dataloader.path}, split: {split.value}")
    print(labels.value_counts())
