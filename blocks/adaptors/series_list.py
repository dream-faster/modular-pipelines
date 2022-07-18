from typing import List
from .base import Adaptor
import pandas as pd
from type import DataType


class SeriesToList(Adaptor):

    inputTypes = DataType.Series
    outputType = DataType.List

    def predict(self, dataset: pd.Series) -> List:
        return dataset.to_list()
