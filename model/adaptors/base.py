from configs.constants import Const
from model.base import Model
import pandas as pd
from type import BaseConfig
from utils.random import random_string
from typing import Optional


class Adaptor(Model):
    def __init__(self, id: Optional[str] = None, config: BaseConfig = None):
        self.config = BaseConfig(force_fit=False) if config is None else config
        self.id = self.__class__.__name__ + "-" + random_string(5) if id is None else id

    def preload(self):
        pass

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        return True
