from typing import List, Optional, Union

import numpy as np
import pandas as pd

from runner.store import Store

from .base import DataSource, Element
from type import BaseConfig, DataType, Hierarchy, SourceTypes
from utils.process_block import process_block


class Concat(Element):
    blocks: List[Union[DataSource, "Pipeline"]]

    def __init__(
        self,
        id: str,
        blocks: List[Union[DataSource, "Pipeline", "Concat"]],
        datasource_labels: DataSource,
    ):
        self.blocks = blocks
        self.id = id
        self.datasource_labels = datasource_labels

    def deplate(self, store: Store, plugins: List["Plugin"], train: bool) -> pd.Series:
        collected = self.transform(
            [process_block(block, store, plugins, train) for block in self.blocks]
        )

        return collected

    def get_labels(self, source_type: Optional[SourceTypes] = None) -> pd.Series:
        return self.datasource_labels.get_labels(source_type)

    def load(self, plugins: List["Plugin"]):
        for block in self.blocks:
            block.load(plugins)

    def transform(self, data: List[pd.Series]) -> pd.Series:
        raise NotImplementedError()

    def children(self, source_type: SourceTypes) -> List[Element]:
        return [self] + [block.children(source_type) for block in self.blocks]

    def get_hierarchy(self, source_type: SourceTypes) -> Hierarchy:
        return Hierarchy(
            name=self.id,
            obj=self,
            children=[block.get_hierarchy(source_type) for block in self.blocks],
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
    outputType = DataType.NpArray

    def transform(self, sources: List[List]) -> np.ndarray:
        return np.hstack([[item[1] for item in data] for data in sources])
