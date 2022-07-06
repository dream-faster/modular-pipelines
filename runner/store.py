import pandas as pd
from typing import Dict
from utils.random import random_string


class Store:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.id = "store-" + random_string(5)

    def get_data(self, id: str) -> pd.DataFrame:
        return self.data[id]

    def set_data(self, id: str, data: pd.DataFrame) -> None:
        self.data[id] = data
