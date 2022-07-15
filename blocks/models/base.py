from type import BaseConfig
import pandas as pd
from typing import Optional
from blocks.base import Block
from blocks.iomanager import safe_loading, safe_saving


class Model(Block):

    config: BaseConfig

    def __init__(self, id: str, config: BaseConfig) -> None:
        raise NotImplementedError()

    def load(self, pipeline_id: str) -> None:
        model = safe_loading(pipeline_id, self.id)
        if model is not None:
            self.model = model

    def load_remote(self) -> None:
        pass

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        raise NotImplementedError()

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        raise NotImplementedError()

    def save(self, pipeline_id: str) -> None:
        if hasattr(self, "trained") and self.trained:
            safe_saving(self.model, pipeline_id, self.id)

    def save_remote(self, pipeline_id: str) -> None:
        pass
