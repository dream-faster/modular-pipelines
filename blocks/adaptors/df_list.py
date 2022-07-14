from typing import List
from .base import Adaptor
import pandas as pd
import numpy as np
from configs.constants import Const


class DfToList(Adaptor):
    def predict(self, dataset: pd.DataFrame) -> List:
        return dataset.to_list()
