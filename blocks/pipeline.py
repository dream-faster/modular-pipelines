from .base import Block, DataSource, Element
import pandas as pd
from typing import List, Union, Optional, Callable
from runner.train import train_predict, predict
from runner.store import Store


class Pipeline(Block):

    id: str
    datasource: Union[DataSource, "Pipeline"]
    models: List[Block]

    def __init__(
        self,
        id: str,
        datasource: Union[DataSource, "Pipeline"],
        models: Union[List[Block], Block],
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

    def load(self, plugins: List["Plugin"]) -> None:
        """Begin"""
        for plugin in plugins:
            plugin.on_load_begin()

        """ Core """
        last_i = 0
        if isinstance(self.datasource, Pipeline):
            last_i = self.datasource.load(f"{self.datasource.id}/{self.id}", 0)

        for i, model in enumerate(self.models):
            model.load(f"{self.datasource.id}/{self.id}", i + last_i)

        """ End """
        for plugin in plugins:
            plugin.on_load_end()

    def fit(self, store: Store, plugins: List["Plugin"]) -> None:
        """Begin"""
        last_output = process_block(self.datasource, store)
        for plugin in plugins:
            store, last_output = plugin.on_fit_begin(store, last_output)

        """ Core """
        for model in self.models:
            last_output = train_predict(model, last_output, store)

        """ End """
        for plugin in plugins:
            store, last_output = plugin.on_fit_end(store, last_output)

        """ Save data """
        store.set_data(self.id, last_output)

    def predict(self, store: Store, plugins: List["Plugin"]) -> pd.Series:
        """Begin"""
        last_output = process_block(self.datasource, store)
        for plugin in plugins:
            store, last_output = plugin.on_predict_begin(store, last_output)

        """ Core """
        for model in self.models:
            last_output = predict(model, last_output, store)

        """ End """
        for plugin in plugins:
            store, last_output = plugin.on_predict_end(store, last_output)

        store.set_data(self.id, last_output)
        return last_output

    def is_fitted(self) -> bool:
        return all([model.is_fitted() for model in self.models])

    def save(self, plugins: List["Plugin"]) -> None:
        pass

    def save_remote(self) -> None:
        for model in self.models:
            if model.config.save and model.config.save_remote:
                model.save_remote()

    def children(self) -> List[Element]:
        return self.datasource.children() + [self] + [self.models]


def process_block(block: Union[DataSource, Pipeline], store: Store) -> pd.Series:
    if isinstance(block, DataSource):
        return block.deplate(store)
    elif isinstance(block, Pipeline):
        if not block.is_fitted():
            block.fit(store)
        return block.predict(store)
    else:
        raise ValueError("Expected DataBlock or Pipeline")
