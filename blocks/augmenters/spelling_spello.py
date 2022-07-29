from typing import Optional

import pandas as pd
from spello.model import SpellCorrectionModel

from configs import Const
from type import DataType

from .base import Augmenter


class SpellingSpelloAugmenter(Augmenter):

    inputTypes = DataType.List
    outputType = DataType.List

    def load(self) -> None:
        self.sp = SpellCorrectionModel(language="en")

    def predict(self, dataset: pd.Series) -> pd.Series:
        return dataset.apply(self.sp)
