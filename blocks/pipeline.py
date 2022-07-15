from .base import Block, DataSource
from blocks.models.base import Model
import pandas as pd
from typing import List, Union, Optional
from runner.train import train_predict, predict
from runner.store import Store


class Pipeline(Block):

    id: str
    datasource: Union[DataSource, "Pipeline"]
    models: List[Model]

    def __init__(
        self,
        id: str,
        datasource: Union[DataSource, "Pipeline"],
        models: Union[List[Model], Model],
    ):
        self.id = id
        if isinstance(models, list):
            self.models = models
        else:
            self.models = [models]
        self.datasource = datasource

    def load_remote(self):
        self.datasource.load_remote()
        for model in self.models:
            model.load_remote()

    def load(self) -> None:
        for model in self.models:
            model.load(self.id)

    def fit(self, store: Store) -> None:
        last_output = process_block(self.datasource, store)
        for model in self.models:
            last_output = train_predict(model, last_output, store)
        store.set_data(self.id, last_output)

    def predict(self, store: Store) -> pd.DataFrame:
        last_output = process_block(self.datasource, store)
        for model in self.models:
            last_output = predict(model, last_output)
        store.set_data(self.id, last_output)
        return last_output

    def is_fitted(self) -> bool:
        return all([model.is_fitted() for model in self.models])

    def save(self) -> None:
        for model in self.models:
            model.save(self.id)

    def save_remote(self) -> None:
        for model in self.models:
            model.save_remote(self.id)


def process_block(block: Union[DataSource, Pipeline], store: Store) -> pd.DataFrame:
    if isinstance(block, DataSource):
        return block.deplate(store)
    elif isinstance(block, Pipeline):
        if not block.is_fitted():
            block.fit(store)
        return block.predict(store)
    else:
        raise ValueError("Expected DataBlock or Pipeline")
