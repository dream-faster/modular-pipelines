from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from runner.store import Store
from runner.train import predict, train_predict
from type import BaseConfig, Hierarchy
from utils.list import flatten
from utils.printing import logger

from .base import Block, DataSource, Element
from .concat import Concat
from type import DataType, SourceTypes

from utils.process_block import process_block
from constants import Const


class Pipeline(Block):

    id: str
    datasource: Union[DataSource, "Pipeline", "Concat"]
    datasource_predict: Union[DataSource, "Pipeline", "Concat"]
    models: List[Block]

    def __init__(
        self,
        id: str,
        datasource: Union[DataSource, "Pipeline", "Concat"],
        models: Union[List[Block], Block],
        datasource_predict: Optional[Union[DataSource, "Pipeline", "Concat"]] = None,
    ):
        self.id = self.__class__.__name__ if id is None else id
        if isinstance(models, List):
            self.models = [deepcopy(model) for model in models]
        else:
            self.models = [deepcopy(models)]

        self.datasource = datasource

        if datasource_predict is None:
            self.datasource_predict = datasource
        else:
            self.datasource_predict = datasource_predict

    def load(self, plugins: List["Plugin"]) -> None:
        """Begin"""
        for plugin in plugins:
            plugin.print_me("on_load_begin")
            plugin.on_load_begin()

        """ Core """
        if isinstance(self.datasource, Pipeline) or isinstance(self.datasource, Concat):
            self.datasource.load(plugins)
        if self.datasource_predict is not self.datasource:
            if isinstance(self.datasource_predict, Pipeline) or isinstance(
                self.datasource_predict, Concat
            ):
                self.datasource_predict.load(plugins)

        for model in self.models:
            model.load()

        """ End """
        for plugin in plugins:
            plugin.print_me("on_load_end")
            plugin.on_load_end()

    def fit(self, store: Store, plugins: List["Plugin"]) -> None:
        logger.log(f"Training on {self.id}", level=logger.levels.ONE)
        datasource = self.datasource

        """Begin"""
        last_output = process_block(datasource, store, plugins, train=True)
        for plugin in plugins:
            plugin.print_me("on_fit_begin")
            store, last_output = plugin.on_fit_begin(store, last_output)

        """ Core """
        for model in self.models:
            last_output = train_predict(
                model, last_output, datasource.get_labels(), store
            )

        """ End """
        for plugin in plugins:
            plugin.print_me("on_fit_end")
            store, last_output = plugin.on_fit_end(store, last_output)

        """ Save data """
        store.set_data(self.id, last_output)

    def predict(self, store: Store, plugins: List["Plugin"]) -> pd.Series:
        logger.log(f"Predicting on {self.id}", level=logger.levels.ONE)
        datasource = self.datasource_predict

        """Begin"""
        last_output = process_block(datasource, store, plugins, train=False)
        for plugin in plugins:
            plugin.print_me("on_predict_begin")
            store, last_output = plugin.on_predict_begin(store, last_output)

        """ Core """
        for model in self.models:
            last_output = predict(model, last_output, datasource.get_labels(), store)

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

    def children(self, source_type: SourceTypes) -> List[Element]:
        if source_type == SourceTypes.fit:
            return self.datasource.children(source_type) + [self] + [self.models]
        elif source_type == SourceTypes.predict:
            return (
                self.datasource_predict.children(source_type) + [self] + [self.models]
            )
        else:
            raise KeyError(f"source-type: {source_type} not a valid type")

    def get_hierarchy(self, source_type: SourceTypes) -> Hierarchy:
        if source_type == SourceTypes.fit:
            return get_source_hierarchy(
                self, self.datasource.get_hierarchy(source_type)
            )
        elif source_type == SourceTypes.predict:
            return get_source_hierarchy(
                self, self.datasource_predict.get_hierarchy(source_type)
            )

    def get_configs(self, source_type: SourceTypes) -> Dict[str, dict]:
        entire_pipeline = self.children(source_type)
        return {
            block.id: vars(block.config)
            for block in flatten(entire_pipeline)
            if not any(
                [
                    isinstance(block, DataSource),
                    isinstance(block, Pipeline),
                    isinstance(block, Concat),
                ]
            )
        }

    def get_datasource_configs(self, source_type: SourceTypes) -> Dict[str, dict]:
        entire_pipeline = self.children(source_type)
        return {
            block.id: vars(block.dataloader.preprocessing_configs[0])
            for block in flatten(entire_pipeline)
            if isinstance(block, DataSource)
        }

    def get_datasource_types(self) -> List[SourceTypes]:
        """
        Checks if any block in the pipeline has
        a different datasource for SourceTypes.predict and Sourcetypes.fit.

        Returns
        -------
        List[str]

        """

        if self.children(SourceTypes.fit) == self.children(SourceTypes.predict):
            return [SourceTypes.fit]
        else:
            return [SourceTypes.fit, SourceTypes.predict]

    def get_labels(self, source_type: SourceTypes) -> pd.Series:
        if source_type == SourceTypes.predict:
            return self.datasource_predict.get_labels(source_type)
        else:
            return self.datasource.get_labels(source_type)


def get_source_hierarchy(pipeline: Pipeline, source_hierarchy: Hierarchy) -> Hierarchy:
    current_pipeline_hierarchy = Hierarchy(
        name=pipeline.id,
        obj=pipeline,
        children=[child.get_hierarchy() for child in pipeline.models]
        if hasattr(pipeline, "models")
        else [],
    )

    if (
        hasattr(source_hierarchy, "children")
        and source_hierarchy.children is not None
        and len(source_hierarchy.children) > 0
    ):

        current_pipeline_hierarchy.children = [
            source_hierarchy
        ] + current_pipeline_hierarchy.children

        return current_pipeline_hierarchy

    else:
        source_hierarchy.children = [current_pipeline_hierarchy]
        return source_hierarchy
