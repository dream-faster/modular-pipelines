from configs.constants import Const
from .base import Block, Model, DataSource
import pandas as pd
from typing import List, Any, Callable
from runner.train import train_model


class Pipeline(Block):

    id: str
    datasource: DataSource
    models: List[Model]

    def __init__(self, id: str, datasource: DataSource, models: List[Model]):
        self.id = id
        self.models = models
        self.datasource = datasource

    def preload(self):
        for model in self.models:
            model.preload()

    def fit(self, get_data: Callable) -> None:
        last_output = self.datasource.deplate(get_data)
        for model in self.models:
            train_model(model, last_output)
            last_output = pd.DataFrame(
                {
                    Const.input_col: model.predict(last_output[Const.input_col]),
                    Const.label_col: train_dataset[Const.label_col],
                }
            )

    def predict(self, get_data: Callable) -> List[Any]:
        last_output = self.datasource.deplate(get_data)
        for model in self.models:
            last_output = model.predict(last_output)
        return last_output

    def is_fitted(self) -> bool:
        return all([model.is_fitted() for model in self.models])
