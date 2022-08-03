import pandas as pd

from .base import Adaptor
from type import DataType, BaseConfig, PredsWithProbs
from typing import Optional, List


class ClassificationOutputAdaptor(Adaptor):
    inputTypes = DataType.PredictionsWithProbs
    outputType = DataType.List

    def __init__(self, select: int):
        super().__init__()
        assert select < 2, "ClassificationOutputAdaptor only supports 0 or 1 for select"

        self.select = select

    def predict(self, dataset: List[PredsWithProbs]) -> DataType.List:
        return [data[self.select] for data in dataset]
