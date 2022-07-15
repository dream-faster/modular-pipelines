from type import BaseConfig
from abc import ABC
import pandas as pd
from typing import Optional
from runner.store import Store


class Block(ABC):
    id: str


class DataSource(Block):

    id: str

    def __init__(self, id: str):
        self.id = id

    def deplate(self, store: Store) -> pd.DataFrame:
        return store.get_data(self.id)

    def load_remote(self):
        pass
