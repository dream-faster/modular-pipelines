from typing import List, Union

import numpy as np
import pandas as pd

from runner.store import Store

from .base import DataSource
from .pipeline import Pipeline, process_block
from .base import Element
from type import Hierarchy


class Concat:

    blocks: List[Union[DataSource, Pipeline, "Concat"]]

    def __init__(self, id: str, blocks: List[Union[DataSource, Pipeline, "Concat"]]):
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
        return [self] + [block.children() for block in self.blocks]

    def get_hierarchy(self) -> Hierarchy:
        return Hierarchy(
            name=self.id,
            obj=self,
            children=[block.get_hierarchy() for block in self.blocks],
        )


class StrConcat(Concat):
    def transform(self, data: List[pd.Series]) -> pd.Series:
        return pd.concat(data, axis=1).agg("-".join, axis=1)


class VectorConcat(Concat):
    def transform(self, data: List[pd.Series]) -> pd.Series:
        return pd.concat(data, axis=1).agg(np.concatenate, axis=1)
