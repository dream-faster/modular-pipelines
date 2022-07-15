from typing import Optional
import pandas as pd
from spello.model import SpellCorrectionModel
from configs import Const
from .base import Augmenter
from type import DataType


class SpellingSpelloAugmenter(Augmenter):

    inputTypes = DataType.List_str
    outputType = DataType.List_str

    def load_remote(self):
        self.sp = SpellCorrectionModel(language="en")

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset[Const.input_col] = dataset[Const.input_col].apply(self.sp)
        return dataset
