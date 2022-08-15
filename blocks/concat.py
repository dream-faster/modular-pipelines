from typing import List, Union

import numpy as np
import pandas as pd

from runner.store import Store

from .base import DataSource, Element
from type import BaseConfig, DataType, Hierarchy
from utils.process_block import process_block


class Concat(Element):
    blocks: List[Union[DataSource, "Pipeline"]]

    def __init__(self, id: str, blocks: List[Union[DataSource, "Pipeline", "Concat"]]):
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
    inputTypes = DataType.List
    outputType = DataType.List

    def transform(self, data: List[List[int]]) -> List:
        return list(zip(*data))


class ClassificationOutputConcat(Concat):
    inputTypes = DataType.PredictionsWithProbs
    outputType = DataType.List

    def transform(self, sources: List[List]) -> List:
        return list(zip(*[data[1] for data in sources]))
