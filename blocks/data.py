from typing import List, Union

import numpy as np
import pandas as pd

from configs.constants import Const
from runner.store import Store

from .base import DataSource
from .pipeline import Pipeline, process_block
from .base import Element


class BaseConcat(DataSource):

    blocks: List[Union[DataSource, Pipeline]]

    def __init__(self, id: str, blocks: List[Union[DataSource, Pipeline]]):
        self.blocks = blocks
        self.id = id

    def deplate(self, store: Store, plugins: List["Plugin"]) -> pd.DataFrame:
        collected = self.transform(
            [process_block(block, store, plugins) for block in self.blocks]
        )

        return collected

    def load(self, plugins: List["Plugin"]):
        for block in self.blocks:
            block.load(plugins)

    def transform(self, data: List[pd.Series]) -> pd.Series:
        raise NotImplementedError()

    def children(self) -> List[Element]:
        return [self] + [self.blocks]

    def dict_children(self) -> dict:
        return {"name": self.id, "obj": self, "children": self.blocks}


class StrConcat(BaseConcat):
    def transform(self, data: List[pd.Series]) -> pd.Series:
        return pd.concat(data, axis=1).agg("-".join, axis=1)


class VectorConcat(BaseConcat):
    def transform(self, data: List[pd.Series]) -> pd.Series:
        return pd.concat(data, axis=1).agg(np.concatenate, axis=1)
