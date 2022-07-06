from .base import DataSource
import pandas as pd
from typing import Callable, List, Union
from .pipeline import Pipeline


class Concat(DataSource):
    def __init__(self, pipelines_or_datablocks: List[Union[Pipeline, DataSource]]):
        self.pipelines_or_datablocks = pipelines_or_datablocks

    def deplate(self, get_data: Callable) -> pd.DataFrame:
        collected = [
            process_datablock_or_pipeline(block, get_data)
            for block in self.pipelines_or_datablocks
        ]
        return pd.DataFrame(collected)


def process_datablock_or_pipeline(
    block: Union[DataSource, Pipeline], get_data: Callable
) -> pd.DataFrame:
    if isinstance(block, DataSource):
        return block.deplate(get_data)
    elif isinstance(block, Pipeline):
        return block.predict(get_data)
    else:
        raise ValueError("Expected DataBlock or Pipeline")
