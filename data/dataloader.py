from type import (
    PreprocessConfig,
    DatasetCategories,
    TrainDataset,
    TestDataset,
)
from typing import List, Callable, Tuple, Optional, Union
from .transformation import transform_dataset
from datasets.load import load_dataset

from .merge import merge_datasets


class DataLoaderBase:
    def transform_(self) -> None:
        pass

    def load(self, category: DatasetCategories) -> Union[TrainDataset, TestDataset]:
        pass


class DataLoader(DataLoaderBase):
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

    def transform_(self) -> None:
        for transformation in self.transformations:
            self.data = transformation(self.data, self.preprocessing_config)

    def load(self, category: DatasetCategories) -> Union[TrainDataset, TestDataset]:
        return self.data[category]


class DataLoaderMerger(DataLoaderBase):
    def __init__(self, data_loaders=List[DataLoader]):
        self.data_loaders = data_loaders
        self.data = []
        pass

    def load(self, category: DatasetCategories) -> Union[TrainDataset, TestDataset]:
        return merge_datasets(
            [
                data_loader.transform_().load(category)
                for data_loader in self.data_loaders
            ]
        )
