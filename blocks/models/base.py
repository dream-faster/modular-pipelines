from type import BaseConfig
import pandas as pd
from typing import Optional
from blocks.base import Block


class Model(Block):

    config: BaseConfig

    def __init__(self, id: Optional[str] = None) -> None:
        pass

    def load(self, pipeline_id: str, execution_order: int) -> None:
        pass

    def load_remote(self) -> None:
        pass

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        raise NotImplementedError()

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        raise NotImplementedError()

    def save(self, pipeline_id: str) -> None:
        pass

    def save_remote(self, pipeline_id: str) -> None:
        pass
