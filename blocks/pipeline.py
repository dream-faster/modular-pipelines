from copy import deepcopy
from typing import Callable, List, Optional, Union

import pandas as pd

from configs.constants import LogConst
from runner.store import Store
from runner.train import predict, train_predict
from type import BaseConfig, Hierarchy
from utils.flatten import flatten

from .base import Block, DataSource, Element


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
        if isinstance(models, List):
            self.models = [deepcopy(model) for model in models]
        else:
            self.models = [deepcopy(models)]
        self.datasource = datasource

    def load(self, plugins: List["Plugin"]) -> None:
        """Begin"""
        for plugin in plugins:
            plugin.print_me("on_load_begin")
            plugin.on_load_begin()

        """ Core """
        if isinstance(self.datasource, Pipeline):
            self.datasource.load()

        for model in self.models:
            model.load()

        """ End """
        for plugin in plugins:
            plugin.print_me("on_load_end")
            plugin.on_load_end()

    def fit(self, store: Store, plugins: List["Plugin"]) -> None:
        """Begin"""
        last_output = process_block(self.datasource, store, plugins)
        for plugin in plugins:
            plugin.print_me("on_fit_begin")
            store, last_output = plugin.on_fit_begin(store, last_output)

        """ Core """
        for model in self.models:
            last_output = train_predict(model, last_output, store)

        """ End """
        for plugin in plugins:
            plugin.print_me("on_fit_end")
            store, last_output = plugin.on_fit_end(store, last_output)

        """ Save data """
        store.set_data(self.id, last_output)

    def predict(self, store: Store, plugins: List["Plugin"]) -> pd.Series:
        """Begin"""
        last_output = process_block(self.datasource, store, plugins)
        for plugin in plugins:
            plugin.print_me("on_predict_begin")
            store, last_output = plugin.on_predict_begin(store, last_output)

        """ Core """
        for model in self.models:
            last_output = predict(model, last_output, store)

        """ End """
        for plugin in plugins:
            plugin.print_me("on_predict_end")
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

    def get_hierarchy(self) -> dict:
        source_hierarchy = self.datasource.get_hierarchy()

        if not hasattr(source_hierarchy, "children"):
            if len(source_hierarchy.children == 0):
                source_hierarchy.children = [
                    Hierarchy(
                        name=self.id,
                        obj=self,
                        children=[child.get_hierarchy() for child in self.models]
                        if hasattr(self, "models")
                        else [],
                    )
                ]

        return source_hierarchy

    def get_configs(self) -> List[BaseConfig]:
        entire_pipeline = self.children()
        return {
            block.id: vars(block.config)
            for block in flatten(entire_pipeline)
            if not any([isinstance(block, DataSource), isinstance(block, Pipeline)])
        }


def process_block(
    block: Union[DataSource, Pipeline],
    store: Store,
    plugins: List["Plugin"],
) -> pd.Series:
    if isinstance(block, DataSource):
        return block.deplate(store, plugins)
    elif isinstance(block, Pipeline):
        if not block.is_fitted():
            block.fit(store, plugins)
        return block.predict(store, plugins)
    else:
        raise ValueError("Expected DataBlock or Pipeline")
