from abc import ABC
from typing import Callable, List, Tuple, Any

from blocks.pipeline import Pipeline

from utils.random import random_string
from runner.store import Store
import pandas as pd


def print_checker(function_origin: str, text: str) -> None:
    if not function_origin == "Plugin":
        print(text)


class Plugin(ABC):
    def __init__(self):
        self.id = f"{self.__class__.__name__} - {random_string(5)}"
        self.printprefix = f"    ┣━━┯ 🔌 Plugin {self.id}: "

    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        function_origin = type(self).on_run_begin.__qualname__.split(".")[0]
        print_checker(function_origin, f"{self.printprefix} on_run_begin")

        return pipeline

    def on_load_begin(self):
        function_origin = type(self).on_load_begin.__qualname__.split(".")[0]
        print_checker(function_origin, f"{self.printprefix} on_load_begin")

    def on_load_end(self):
        function_origin = type(self).on_load_end.__qualname__.split(".")[0]
        print_checker(function_origin, f"{self.printprefix} on_load_end")

    def on_fit_begin(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        function_origin = type(self).on_fit_begin.__qualname__.split(".")[0]
        print_checker(function_origin, f"{self.printprefix} on_fit_begin")

        return store, last_output

    def on_fit_end(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        function_origin = type(self).on_fit_end.__qualname__.split(".")[0]
        print_checker(function_origin, f"{self.printprefix} on_fit_end")

        return store, last_output

    def on_predict_begin(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        function_origin = type(self).on_predict_begin.__qualname__.split(".")[0]
        print_checker(function_origin, f"{self.printprefix} on_predict_begin")

        return store, last_output

    def on_predict_end(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        function_origin = type(self).on_predict_end.__qualname__.split(".")[0]
        print_checker(function_origin, f"{self.printprefix} on_predict_end")

        return store, last_output

    def on_run_end(self, pipeline: Pipeline, stats: pd.Series):
        function_origin = type(self).on_run_end.__qualname__.split(".")[0]
        print_checker(function_origin, f"{self.printprefix} on_run_end")
