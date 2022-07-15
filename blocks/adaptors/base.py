from blocks.base import Block
import pandas as pd
from type import BaseConfig
from typing import Optional


class Adaptor(Block):
    def __init__(self, id: Optional[str] = None, config: Optional[BaseConfig] = None):
        super().__init__()
        self.config = BaseConfig(force_fit=False) if config is None else config

    def load_remote(self):
        pass

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        return True
