from configs.constants import Const
from blocks.base import Block
import pandas as pd
from type import BaseConfig
from utils.random import random_string
from typing import Optional


class Adaptor(Block):
    def load_remote(self):
        pass

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        return True
