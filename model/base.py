from type import BaseConfig
from abc import ABC
import pandas as pd
from typing import List, Tuple
from type import Label, Probabilities


class BaseModel(ABC):

    config: BaseConfig

    def __init__(self, config: BaseConfig):
        raise NotImplementedError()

    def fit(self, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame) -> None:
        raise NotImplementedError()

    def predict(self, test_dataset: pd.DataFrame) -> List[Tuple[Label, Probabilities]]:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        raise NotImplementedError()
