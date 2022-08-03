import pandas as pd

from blocks.base import Block
from type import DataType
from typing import Optional


class ClassificationOutputAdaptor(Block):
    inputTypes = DataType.PredictionsWithProbs
    outputType = DataType.List

    def __init__(self, select: int, id: Optional[str] = None):
        assert select < 2, "ClassificationOutputAdaptor only supports direction 0 or 1"

        self.id = self.__class__.__name__ if id is None else id
        self.select = select

    def predict(self, dataset: DataType.PredictionsWithProbs) -> pd.Series:
        return dataset[self.select]
