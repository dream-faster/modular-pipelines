from configs.constants import Const
from model.base import Model
import pandas as pd
from type import BaseConfig
from utils.random import random_string


class Transformation(Model):
    def __init__(self):
        self.config = BaseConfig(force_fit=False)
        self.id = self.__class__.__name__ + "-" + random_string(5)

    def preload(self):
        pass

    def fit(self, dataset: pd.DataFrame):
        pass

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        return True
