import pandas as pd

from .base import Adaptor
from type import DataType
from typing import Optional


class ClassificationOutputAdaptor(Adaptor):
    inputTypes = DataType.PredictionsWithProbs
    outputType = DataType.Series

    def __init__(self, select: int, id: Optional[str] = None):
        assert select < 2, "ClassificationOutputAdaptor only supports direction 0 or 1"

        self.id = self.__class__.__name__ if id is None else id
        self.select = select

    def predict(self, dataset: DataType.PredictionsWithProbs) -> DataType.Series:
        return dataset[self.select]
