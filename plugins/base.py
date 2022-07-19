from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Any

from utils.random import random_string
from runner.store import Store


class Plugin(ABC):
    def __init__(self):
        self.id = random_string(5)

    def on_fit_begin(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        print(f"On Fit Begin for: {self.id}")

        return store, last_output

    def on_fit_end(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        print(f"On Fit End for: {self.id}")

        return store, last_output
