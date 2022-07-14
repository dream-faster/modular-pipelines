from type import BaseConfig
import pandas as pd
from typing import Optional
from blocks.base import Block


class Model(Block):

    config: BaseConfig

    def __init__(self, id: str, config: BaseConfig):
        raise NotImplementedError()

    def preload(self):
        pass

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        raise NotImplementedError()

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        raise NotImplementedError()
