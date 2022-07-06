from configs.constants import Const
from model.base import Model
from typing import List, Any
import pandas as pd

from type import BaseConfig


class IdentityAugmenter(Model):
    def __init__(self):
        self.config = BaseConfig(force_fit=False)

    def preload(self):
        pass

    def fit(self, train_dataset: pd.DataFrame):
        pass

    def predict(self, test_dataset: pd.DataFrame) -> pd.DataFrame:
        return test_dataset

    def is_fitted(self) -> bool:
        return True
