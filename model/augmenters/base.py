from model.base import Model
import pandas as pd

from type import BaseConfig


class Augmenter(Model):
    def __init__(self, id: str):
        self.config = BaseConfig(force_fit=False)
        self.id = id

    def preload(self):
        pass

    def fit(self, dataset: pd.DataFrame):
        pass

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        return True
