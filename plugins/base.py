from abc import ABC
from typing import Callable, List, Tuple, Any

from blocks.pipeline import Pipeline
from configs.constants import LogConst

from utils.random import random_string
from runner.store import Store
import pandas as pd
import logging


def just_custom_functions(obj) -> List[Tuple]:
    return [func for func in vars(obj).items() if not func[0].startswith("__")]


def get_parent_name(class_obj) -> List[str]:
    return [
        parent.__name__
        for parent in class_obj.__mro__
        if parent.__name__ not in ["ABC", "object"]
    ][::-1]


class Plugin(ABC):
    def __init__(self):
        pass

    def __init_subclass__(cls):
        cls.id = f"{cls.__name__} - {random_string(5)}"
        cls.print_dict = vars(cls).keys()

        cls.logger = logging.getLogger(".".join(get_parent_name(cls)))
        cls.d = {"lines": f"{'    '*len(get_parent_name(cls))}┃", "splits": "  └──"}

    def print_me(self, key):
        if key in self.print_dict:
            d = {"lines": f"{'    '*len(get_parent_name(type(self)))}┃", "splits": ""}
            self.logger.info(
                f"{LogConst.indentation}{LogConst.plugin_prefix} {self.id}: {key}",
                extra=d,
            )

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

    def on_save_remote_begin(self) -> None:
        pass

    def on_save_remote_end(self) -> None:
        pass

    def on_run_end(
        self, pipeline: Pipeline, stats: pd.Series
    ) -> Tuple[Pipeline, pd.Series]:
        return pipeline, stats
