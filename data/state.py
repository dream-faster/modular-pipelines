import pandas as pd


class GlobalState:
    def __init__(self):
        self.state = dict()

    def get(self, key: str):
        return self.state[key]

    def set(self, key: str, value: pd.DataFrame) -> None:
        self.state[key] = value
