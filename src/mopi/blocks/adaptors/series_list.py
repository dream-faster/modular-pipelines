from typing import List

import pandas as pd

from mopi.type import DataType

from .base import Adaptor


class SeriesToList(Adaptor):

    inputTypes = DataType.Series
    outputType = DataType.List

    def predict(self, dataset: pd.Series) -> List:
        return dataset.to_list()
