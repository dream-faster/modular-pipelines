from .base import Block, Model, DataSource
import pandas as pd
from typing import List, Union
from runner.train import train_predict
from runner.store import Store


class Pipeline(Block):

    id: str
    datasource: DataSource
    models: List[Model]

    def __init__(
        self, id: str, datasource: DataSource, models: Union[List[Model], Model]
    ):
        self.id = id
        if isinstance(models, list):
            self.models = models
        else:
            self.models = [models]
        self.datasource = datasource

    def preload(self):
        self.datasource.preload()
        for model in self.models:
            model.preload()

    def fit(self, store: Store) -> None:
        last_output = self.datasource.deplate(store)
        for model in self.models:
            last_output = train_predict(model, last_output, store)
        store.set_data(self.id, last_output)

    def predict(self, store: Store) -> pd.DataFrame:
        last_output = self.datasource.deplate(store)
        for model in self.models:
            last_output = model.predict(last_output)
        store.set_data(self.id, last_output)
        return last_output

    def is_fitted(self) -> bool:
        return all([model.is_fitted() for model in self.models])
