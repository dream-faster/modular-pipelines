from abc import ABC
from typing import Any, Callable, List, Tuple

import pandas as pd

from mopi.blocks.pipeline import Pipeline
from mopi.constants import LogConst
from mopi.runner.store import Store
from mopi.utils.random import random_string
from mopi.utils.printing import logger


def just_custom_functions(obj) -> List[Tuple]:
    return [func for func in vars(obj).items() if not func[0].startswith("__")]


class Plugin(ABC):
    def __init__(self):
        pass

    def __init_subclass__(cls):
        cls.id = f"{cls.__name__} - {random_string(5)}"
        cls.print_dict = vars(cls).keys()

    def print_me(self, key):
        if key in self.print_dict:
            logger.log(
                f"{LogConst.plugin_prefix} {self.id}: {key}", level=logger.levels.ONE
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

    def on_run_end(self, pipeline: Pipeline, store: Store) -> Tuple[Pipeline, Store]:
        return pipeline, store
