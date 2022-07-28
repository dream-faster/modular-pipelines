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
        self.pipeline_id = pipeline_id
        self.id += f"-{str(execution_order)}"

        self.sp = SpellCorrectionModel(language="en")

        return execution_order + 1

    def predict(self, dataset: pd.Series) -> pd.Series:
        return dataset.apply(self.sp)
