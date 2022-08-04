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


class DataLoader:
    def __init__(
        self,
        path: str,
        preprocessing_config: PreprocessConfig,
        name: Optional[str] = None,
        transformations: List[Callable] = [],
    ):
        obligatory_transformations = [transform_dataset]

        self.transformations = obligatory_transformations + transformations
        self.preprocessing_config = preprocessing_config
        self.data = load_dataset(path, name)

    def load(self, category: DatasetSplit) -> Union[TrainDataset, TestDataset]:
        for transformation in self.transformations:
            self.data = transformation(self.data, self.preprocessing_config)
        return self.data[category.value]


class DataLoaderMerger(DataLoader):
    def __init__(self, data_loaders=List[DataLoader]):
        self.data_loaders = data_loaders

    def load(self, category: DatasetSplit) -> Union[TrainDataset, TestDataset]:
        return merge_datasets(
            [data_loader.load(category) for data_loader in self.data_loaders]
        )
