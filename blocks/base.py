from abc import ABC
from typing import Callable, List, Optional, Union

import pandas as pd

from blocks.io import pickle_loading, pickle_saving
from runner.store import Store
from type import BaseConfig, DataType, Hierarchy, RunContext
from configs.constants import Const
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

    def __init__(self, id: str):
        self.id = id

    def deplate(self, store: Store, plugins: List["Plugin"], train: bool) -> pd.Series:
        return store.get_data(self.id)

    def load(self) -> None:
        pass

    def children(self) -> List[Element]:
        return [self]

    def get_hierarchy(self) -> Hierarchy:
        return Hierarchy(name=self.id, obj=self)
