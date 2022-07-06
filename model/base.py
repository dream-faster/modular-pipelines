from type import BaseConfig
from abc import ABC
import pandas as pd
from typing import Any, List, Tuple
from type import Label, Probabilities


class BaseModel(ABC):

    config: BaseConfig

    def __init__(self, config: BaseConfig):
        raise NotImplementedError()

    def preload(self):
        pass

    def fit(self, train_dataset: pd.DataFrame) -> None:
        raise NotImplementedError()

    def predict(self, test_dataset: List[Any]) -> List[Any]:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        raise NotImplementedError()

    def get_parent_class(self) -> type:
        for base in self.__class__.__bases__:
            return base
