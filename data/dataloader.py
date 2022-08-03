from type import (
    PreprocessConfig,
    DatasetCategories,
    TrainDataset,
    TestDataset,
)
from typing import List, Callable, Tuple
from .transformation import transform_dataset
from datasets.load import load_dataset


class DataLoader:
    def __init__(
        self,
        preprocessing_config: PreprocessConfig,
        path: str,
        name: str,
        transformations: List[Callable] = [],
    ):
        obligatory_transformations = [transform_dataset]

        self.transformations = obligatory_transformations + transformations
        self.preprocessing_config = preprocessing_config
        self.name = name
        self.path = path

    def load(self, category: DatasetCategories) -> Tuple[TrainDataset, TestDataset]:
        data = load_dataset(self.path, self.name)

        for transformation in self.transformations:
            data = transformation(data, self.preprocessing_config)

        return data[category]
