from .base import Adaptor
import numpy as np
from type import DataType
from typing import List


class ListOfListsToNumpy(Adaptor):

    inputTypes = DataType.List
    outputType = DataType.NpArray

    def __init__(self, replace_nan: bool) -> None:
        super().__init__()
        self.replace_nan = replace_nan

    def predict(self, dataset: List) -> np.ndarray:
        assert len(dataset) > 0, "Dataset is empty"

        if isinstance(dataset[0], (List or np.ndarray)):
            return np.nan_to_num(np.vstack(dataset), nan=0.0) if self.replace_nan else np.vstack(dataset)
        elif isinstance(dataset[0], (int or float)):
            return np.nan_to_num(np.array(dataset), nan=0.0) if self.replace_nan else np.vstack(dataset)
        else:
            assert False, "Unsupported conversion type"
