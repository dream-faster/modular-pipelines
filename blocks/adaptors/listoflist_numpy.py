from .base import Adaptor
import numpy as np
from configs.constants import Const
from type import DataType
from typing import List


class ListOfListsToNumpy(Adaptor):

    inputTypes = DataType.List
    outputType = DataType.NpArray

    def predict(self, dataset: List) -> np.ndarray:
        assert len(dataset) > 0, "Dataset is empty"

        if isinstance(dataset[0], (list or np.ndarray)):
            return np.vstack(dataset)
        elif isinstance(dataset[0], (int or float)):
            return np.array(dataset)
        else:
            assert False, "Unsupported conversion type"
