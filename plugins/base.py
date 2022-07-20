from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Any

from utils.random import random_string
from runner.store import Store


class Plugin(ABC):
    def __init__(self):
        self.id = f"{self.__class__.__name__} - {random_string(5)}"
        self.printprefix = f"    ├ 🔌 Plugin {self.id}: "

    def on_load_begin(self):
        print(f"{self.printprefix} on_load_begin")

    def on_load_end(self):
        print(f"{self.printprefix} on_load_end")

    def on_fit_begin(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        print(f"{self.printprefix} on_fit_begin")

        return store, last_output

    def on_fit_end(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        print(f"{self.printprefix} on_fit_end")

        return store, last_output

    def on_predict_begin(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        print(f"{self.printprefix} on_predict_begin")

        return store, last_output

    def on_predict_end(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        print(f"{self.printprefix} on_predict_end")

        return store, last_output
