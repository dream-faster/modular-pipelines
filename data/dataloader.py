from type import (
    PreprocessConfig,
    DatasetSplit,
    TrainDataset,
    TestDataset,
)
from typing import List, Callable, Tuple, Optional, Union
from .transformation import transform_dataset
from datasets.load import load_dataset

from .merge import merge_datasets
from utils.flatten import flatten


class DataLoader:

    preprocessing_configs: List[PreprocessConfig]
    is_transformed: bool

    def __init__(
        self,
        path: str,
        preprocessing_config: PreprocessConfig,
        transformation: Callable,
        name: Optional[str] = None,
    ):
        self.transformation = transformation
        self.preprocessing_configs = [preprocessing_config]
        self.data = load_dataset(path, name)
        self.is_transformed = False

    def load(self, category: DatasetSplit) -> Union[TrainDataset, TestDataset]:
        if self.is_transformed == False:
            self.data = self.transformation(self.data, self.preprocessing_configs[0])
            self.is_transformed = True
        return self.data[category.value]


class DataLoaderMerger(DataLoader):
    def __init__(self, dataloaders=List[DataLoader]):
        self.dataloaders = dataloaders
        self.preprocessing_configs = flatten(
            [dataloader.preprocessing_configs for dataloader in dataloaders]
        )

    def load(self, category: DatasetSplit) -> Union[TrainDataset, TestDataset]:
        return merge_datasets(
            [data_loader.load(category) for data_loader in self.dataloaders]
        )
