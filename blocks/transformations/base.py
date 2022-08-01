from typing import List, Optional

import pandas as pd

from blocks.base import Block, Element
from type import BaseConfig, Hierarchy


class Transformation(Block):
    def fit(self, dataset: pd.Series, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.Series) -> pd.Series:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        return True

    def children(self) -> List[Element]:
        return [self]

    def get_hierarchy(self) -> Hierarchy:
        return Hierarchy(name=self.id, obj=self)
