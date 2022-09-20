from mopi.utils.printing import logger
from mopi.type import (
    PreprocessConfig,
    DatasetSplit,
    TrainDataset,
    TestDataset,
)
from mopi.constants import Const
from typing import List, Callable, Optional, Union
from datasets.load import load_dataset

from mopi.utils.list import flatten
import pandas as pd
from imblearn.base import BaseSampler
from mopi.utils.setter import Settable
import numpy as np
from mopi.constants import Const


class DataLoader(Settable):

    preprocessing_config: PreprocessConfig
    sampler: Optional[BaseSampler]
    path: str

    def load(self, category: DatasetSplit) -> Union[TrainDataset, TestDataset]:
        raise NotImplementedError()


class HuggingfaceDataLoader(DataLoader):

    is_transformed: bool

    def __init__(
        self,
        path: str,
        preprocessing_config: PreprocessConfig,
        transformation: Callable,
        sampler: Optional[BaseSampler] = None,
        name: Optional[str] = None,
        shuffle_first: Optional[bool] = False,
    ):
        self.transformation = transformation
        self.preprocessing_config = preprocessing_config
        self.is_transformed = False
        self.sampler = sampler
        self.path = path
        self.shuffle_first = shuffle_first
        self.name = name

    def load(self, category: DatasetSplit) -> Union[TrainDataset, TestDataset]:
        self.data = load_dataset(self.path, self.name)
        if self.is_transformed == False:
            if self.shuffle_first:
                logger.log("⚠️ Shuffling Data", level=logger.levels.TWO)
                self.data = self.data.shuffle(Const.seed)
            self.data = self.transformation(self.data, self.preprocessing_config)
            if self.sampler is not None:
                self.data[DatasetSplit.train.value] = apply_sampler(
                    self.data[DatasetSplit.train.value], self.sampler
                )
            self.is_transformed = True
        return self.data[category.value]


class PandasDataLoader(DataLoader):

    is_sampled = False

    def __init__(
        self,
        path: str,
        preprocessing_config: PreprocessConfig,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        sampler: Optional[BaseSampler] = None,
        shuffle_first: Optional[bool] = False,
    ):
        self.path = path
        self.preprocessing_config = preprocessing_config
        self.train_data = train_data
        self.test_data = test_data
        self.is_transformed = False
        self.sampler = sampler
        self.shuffle_first = shuffle_first
        self.transformed = False

    def load(self, category: DatasetSplit) -> Union[TrainDataset, TestDataset]:
        if self.transformed is False and self.shuffle_first:
            logger.log("⚠️ Shuffling Data", level=logger.levels.TWO)
            self.train_data = self.train_data.sample(frac=1, random_state=Const.seed)
            self.test_data = self.test_data.sample(frac=1, random_state=Const.seed)

            self.transformed = True

        if self.sampler is not None and self.is_sampled == False:
            self.train_data = apply_sampler(self.train_data, self.sampler)
            self.is_sampled = True
        if category == DatasetSplit.test:
            return self.test_data
        else:
            return self.train_data


class MergedDataLoader(DataLoader):
    def __init__(self, dataloaders: List[DataLoader], oversample: bool):
        self.dataloaders = dataloaders
        self.preprocessing_config = dataloaders[0].preprocessing_config
        self.path = "/".join([dl.path for dl in dataloaders])
        self.oversample = oversample

    def load(self, category: DatasetSplit) -> Union[TrainDataset, TestDataset]:
        datasets = [data_loader.load(category) for data_loader in self.dataloaders]
        max_dataset_len = max([len(dataset) for dataset in datasets])
        if self.oversample:
            datasets = [
                dataset.sample(n=max_dataset_len, replace=True, random_state=Const.seed)
                for dataset in datasets
            ]
        else:
            datasets = [
                dataset.sample(n=max_dataset_len, random_state=Const.seed)
                for dataset in datasets
            ]

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
