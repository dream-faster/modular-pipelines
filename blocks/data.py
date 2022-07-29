from typing import List, Union

import numpy as np
import pandas as pd

from configs.constants import Const
from runner.store import Store

from .base import DataSource
from .pipeline import Pipeline, process_block


class BaseConcat(DataSource):

    blocks: List[Union[DataSource, Pipeline]]

    def __init__(self, blocks: List[Union[DataSource, Pipeline]]):
        self.blocks = blocks

    def deplate(self, store: Store) -> pd.DataFrame:
        collected = self.transform(
            [process_block(block, store) for block in self.blocks]
        )

        return collected

    def load(self):
        for block in self.blocks:
            block.load()

    def transform(self, data: List[pd.Series]) -> pd.Series:
        raise NotImplementedError()


class StrConcat(BaseConcat):
    def transform(self, data: List[pd.Series]) -> pd.Series:
        return pd.concat(data, axis=1).agg("-".join, axis=1)


class VectorConcat(BaseConcat):
    def transform(self, data: List[pd.Series]) -> pd.Series:
        return pd.concat(data, axis=1).agg(np.concatenate, axis=1)
