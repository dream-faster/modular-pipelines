from blocks.base import Block
import pandas as pd
from type import BaseConfig
from typing import Optional, List


class Transformation(Block):
    def __init__(self, id: Optional[str] = None, config: Optional[BaseConfig] = None):
        super().__init__()
        self.config = BaseConfig(force_fit=False) if config is None else config

    def load_remote(self):
        pass

    def fit(self, dataset: pd.Series, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.Series) -> pd.Series:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        return True

    def children(self) -> List["Element"]:
        return [self]
