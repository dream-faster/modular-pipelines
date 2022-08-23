import pandas as pd

from .base import Adaptor
from type import DataType, PredsWithProbs
from typing import List, Optional


class ClassificationOutputAdaptor(Adaptor):
    inputTypes = DataType.PredictionsWithProbs
    outputType = DataType.List

    def __init__(self, select: int, dict_lookup: Optional[dict] = None):
        super().__init__()
        assert select < 2, "ClassificationOutputAdaptor only supports 0 or 1 for select"

        self.select = select
        self.dict_lookup = dict_lookup

    def predict(self, dataset: List[PredsWithProbs]) -> DataType.List:
        if self.dict_lookup is not None:
            return [
                self.dict_lookup[selected_item]
                for data in dataset
                for selected_item in data[self.select]
            ]
        else:
            return [data[self.select] for data in dataset]
