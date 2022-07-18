from blocks.base import Block, Element
import pandas as pd
from type import BaseConfig
from typing import Optional, List


class Transformation(Block):
    def load_remote(self):
        pass

    def fit(self, dataset: pd.Series, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.Series) -> pd.Series:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        return True

    def children(self) -> List[Element]:
        return [self]
