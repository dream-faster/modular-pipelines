from typing import Optional
import pandas as pd
from spello.model import SpellCorrectionModel
from configs import Const
from .base import Augmenter
from type import DataType


class SpellingSpelloAugmenter(Augmenter):

    inputTypes = DataType.List
    outputType = DataType.List

    def load(self):
        self.sp = SpellCorrectionModel(language="en")

    def predict(self, dataset: pd.Series) -> pd.Series:
        return dataset.apply(self.sp)
