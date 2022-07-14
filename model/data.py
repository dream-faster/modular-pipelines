from .base import DataSource
import pandas as pd
from typing import List, Union
from .pipeline import Pipeline, process_block
from runner.store import Store
from configs.constants import Const
import numpy as np


class BaseConcat(DataSource):

    blocks: List[Union[DataSource, Pipeline]]

    def __init__(self, blocks: List[Union[DataSource, Pipeline]]):
        self.blocks = blocks

    def deplate(self, store: Store) -> pd.DataFrame:
        collected = self.transform(
            [process_block(block, store) for block in self.blocks]
        )

        return pd.DataFrame({Const.input_col: collected})

    def preload(self):
        for block in self.blocks:
            block.preload()

    def transform(self, data: List[pd.DataFrame]) -> pd.Series:
        raise NotImplementedError()


class StrConcat(BaseConcat):
    def transform(self, data: List[pd.DataFrame]) -> pd.Series:
        dataframes = [df[Const.input_col] for df in data]
        return pd.concat(dataframes, axis=1).agg("-".join, axis=1)


class VectorConcat(BaseConcat):
    def transform(self, data: List[pd.DataFrame]) -> pd.Series:
        dataframes = [df[Const.input_col] for df in data]
        return pd.concat(dataframes, axis=1).agg(np.concatenate, axis=1)


