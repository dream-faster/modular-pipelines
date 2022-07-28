from typing import Optional

import pandas as pd
from configs import Const
from spello.model import SpellCorrectionModel
from type import DataType

from .base import Augmenter


class SpellingSpelloAugmenter(Augmenter):

    inputTypes = DataType.List
    outputType = DataType.List

    def load(self, pipeline_id: str, execution_order: int) -> int:
        self.sp = SpellCorrectionModel(language="en")
        return super().load(pipeline_id, execution_order)

    def predict(self, dataset: pd.Series) -> pd.Series:
        return dataset.apply(self.sp)
