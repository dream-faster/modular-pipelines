from type import (
    PreprocessConfig,
    DatasetSplit,
    TrainDataset,
    TestDataset,
)
from configs.constants import Const
from typing import List, Callable, Optional, Union
from datasets.load import load_dataset

from utils.list import flatten
import pandas as pd
from imblearn.base import BaseSampler
from utils.setter import Settable
import numpy as np


class DataLoader(Settable):

    preprocessing_configs: List[PreprocessConfig]
    is_transformed: bool
    sampler: Optional[BaseSampler]

    def __init__(
        self,
        id: str,
        path: str,
        preprocessing_config: PreprocessConfig,
        transformation: Callable,
        sampler: Optional[BaseSampler] = None,
        name: Optional[str] = None,
    ):
        self.id = id
        self.transformation = transformation
        self.preprocessing_configs = [preprocessing_config]
        self.data = load_dataset(path, name)
        self.is_transformed = False
        self.sampler = sampler

    def load(self, category: DatasetSplit) -> Union[TrainDataset, TestDataset]:
        if self.is_transformed == False:
            self.data = self.transformation(self.data, self.preprocessing_configs[0])
            if self.sampler is not None:
                self.data[DatasetSplit.train.value] = apply_sampler(
                    self.data[DatasetSplit.train.value], self.sampler
                )
            self.is_transformed = True
        return self.data[category.value]


class DataLoaderMerger(DataLoader):
    def __init__(self, dataloaders: List[DataLoader]):
        self.dataloaders = dataloaders
        self.preprocessing_configs = flatten(
            [dataloader.preprocessing_configs for dataloader in self.dataloaders]
        )

    def load(self, category: DatasetSplit) -> Union[TrainDataset, TestDataset]:
        datasets = [data_loader.load(category) for data_loader in self.dataloaders]
        return pd.concat(datasets, axis=0).reset_index(drop=True)


def apply_sampler(data: pd.DataFrame, sampler: BaseSampler) -> pd.DataFrame:
    resampled_X, resampled_y = sampler.fit_resample(
        np.array(data[Const.input_col]).reshape(-1, 1),
        np.array(data[Const.label_col]),
    )
    return pd.DataFrame(
        {
            Const.input_col: np.squeeze(resampled_X),
            Const.label_col: resampled_y,
        }
    )
