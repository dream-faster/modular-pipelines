from abc import ABC
from typing import Callable, List, Tuple, Any

from blocks.pipeline import Pipeline

from utils.random import random_string
from runner.store import Store
import pandas as pd


class Plugin(ABC):
    def __init__(self):
        self.id = f"{self.__class__.__name__} - {random_string(5)}"

    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        return pipeline

    def on_load_begin(self) -> None:
        pass

    def on_load_end(self) -> None:
        pass

    def on_fit_begin(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        return store, last_output

    def on_fit_end(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        return store, last_output

    def on_predict_begin(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        return store, last_output

    def on_predict_end(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        return store, last_output

    def on_run_end(self, pipeline: Pipeline, stats: pd.Series):
        return pipeline, stats
