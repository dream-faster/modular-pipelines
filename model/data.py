from abc import ABC
from .base import DataSource
import pandas as pd
from typing import Callable, List, Union
from .pipeline import Pipeline
from runner.store import Store


class BaseConcat(DataSource):

    blocks: List[Union[DataSource, Pipeline]]

    def __init__(self, blocks: List[Union[DataSource, Pipeline]]):
        self.blocks = blocks

    def deplate(self, store: Store) -> pd.DataFrame:
        collected = self.transform(
            [process_block(block, store) for block in self.blocks]
        )
        return pd.DataFrame(collected)

    def preload(self):
        for block in self.blocks:
            block.preload()

    def transform(self, data: List[pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError()


class StrConcat(BaseConcat):
    def transform(self, data: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(data, axis=1)


def process_block(block: Union[DataSource, Pipeline], store: Store) -> pd.DataFrame:
    if isinstance(block, DataSource):
        return block.deplate(store)
    elif isinstance(block, Pipeline):
        if not block.is_fitted():
            block.fit(store)
        return block.predict(store)
    else:
        raise ValueError("Expected DataBlock or Pipeline")
