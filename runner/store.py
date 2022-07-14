import pandas as pd
from typing import Dict
from utils.random import random_string


class Store:
    def __init__(self, data: Dict[str, pd.DataFrame], labels: pd.Series):
        self.data = data
        self.labels = labels
        self.id = "store-" + random_string(5)

    def get_data(self, id: str) -> pd.DataFrame:
        return self.data[id].copy()

    def get_labels(self) -> pd.Series:
        return self.labels.copy()

    def set_data(self, id: str, data: pd.DataFrame) -> None:
        self.data[id] = data
