from type import BaseConfig
from abc import ABC
import pandas as pd
from typing import Any, List, Callable


class Block(ABC):
    id: str


class Model(Block):

    config: BaseConfig

    def __init__(self, config: BaseConfig):
        raise NotImplementedError()

    def preload(self):
        pass

    def fit(self, train_dataset: pd.DataFrame) -> None:
        raise NotImplementedError()

    def predict(self, test_dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        raise NotImplementedError()


class DataSource(Block):

    id: str

    def __init__(self, id: str):
        self.id = id

    def deplate(self, get_data: Callable) -> pd.DataFrame:
        return get_data[self.id]
