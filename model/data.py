from .base import DataSource
import pandas as pd
from typing import Callable, List, Union
from .pipeline import Pipeline
from runner.store import Store


class Concat(DataSource):
    def __init__(self, pipelines_or_datablocks: List[Union[Pipeline, DataSource]]):
        self.pipelines_or_datablocks = pipelines_or_datablocks

    def deplate(self, store: Store) -> pd.DataFrame:
        collected = [
            process_block(block, store) for block in self.pipelines_or_datablocks
        ]
        return pd.DataFrame(collected)

    def preload(self):
        for block in self.pipelines_or_datablocks:
            block.preload()


def process_block(block: Union[DataSource, Pipeline], store: Store) -> pd.DataFrame:
    if isinstance(block, DataSource):
        return block.deplate(store)
    elif isinstance(block, Pipeline):
        if not block.is_fitted():
            block.fit(store)
        return block.predict(store)
    else:
        raise ValueError("Expected DataBlock or Pipeline")
