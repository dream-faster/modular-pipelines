from blocks.models.base import Model
import pandas as pd

from type import BaseConfig
from utils.random import random_string
from typing import Optional


class Augmenter(Model):
    def __init__(self, id: Optional[str] = None, config: BaseConfig = None):
        self.config = BaseConfig(force_fit=False) if config is None else config
        self.id = self.__class__.__name__ if id is None else id

    def load_remote(self):
        pass

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        return True
