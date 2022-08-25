from abc import ABC
from typing import Callable, List, Optional, Union

import pandas as pd

from blocks.io import pickle_loading, pickle_saving
from data.dataloader import DataLoader
from runner.store import Store
from type import BaseConfig, DataType, Hierarchy, RunContext, DatasetSplit, SourceTypes
from constants import Const
from utils.printing import logger


class Element(ABC):
    id: str

    inputTypes: Union[List[DataType], DataType]
    outputType: DataType

    def children(self) -> List["Element"]:
        raise NotImplementedError()


class Block(Element):
    config: BaseConfig
    run_context: RunContext
    parent_path: str

    def __init__(
        self, id: Optional[str] = None, config: Optional[BaseConfig] = None
    ) -> None:
        self.id = self.__class__.__name__ if id is None else id
        self.config = (
            BaseConfig(
                frozen=False,
                save=True,
                save_remote=False,
                preferred_load_origin=None,
            )
            if config is None
            else config
        )

        if self.inputTypes is None:
            logger.log("inputTypes must be set", level=logger.levels.TWO)
        if self.outputType is None:
            logger.log("outputType must be set", level=logger.levels.TWO)

    def fit(
        self,
        dataset: pd.Series,
        labels: Optional[pd.Series],
    ) -> None:
        raise NotImplementedError()

    def predict(self, dataset: pd.Series) -> List:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        raise NotImplementedError()

    def save(self) -> None:
        raise NotImplementedError()

    def load(self) -> None:
        raise NotImplementedError()

    def save_remote(self) -> None:
        pass


class DataSource(Element):

    id: str

    inputTypes = DataType.Any
    outputType = DataType.Series
    dataloader: DataLoader

    def __init__(self, id: str, dataloader: DataLoader) -> None:
        self.id = id
        self.dataloader = dataloader

    def deplate(
        self,
        store: Optional[Store] = None,
        plugins: Optional[List["Plugin"]] = None,
        train: Optional[bool] = None,
    ) -> pd.Series:
        if not hasattr(self, "data") and hasattr(self, "category"):
            self.data = self.dataloader.load(self.category)
            logger.log(
                f"✅ Loaded dataloader on datasource: {self.id}", level=logger.levels.ONE
            )

        return self.data[Const.input_col]

    def load(self, plugins: List["Plugin"]) -> None:
        pass

    def get_labels(self, source_type: Optional[SourceTypes] = None) -> pd.Series:
        return self.data[Const.label_col]

    def children(self, source_type: SourceTypes) -> List[Element]:
        return [self]

    def get_hierarchy(self, source_type: SourceTypes) -> Hierarchy:
        return Hierarchy(name=self.id, obj=self)
