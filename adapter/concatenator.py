from adapter.base import BaseAdapter
from data.state import GlobalState
from typing import List, Optional
import pandas as pd
from configs.constants import Const


class ConcatenatorAdapter(BaseAdapter):
    def __init__(self, name: str, keys: List[str]):
        self.keys = keys
        self.name = name

    def connect(self, state: GlobalState):
        item = pd.DataFrame(columns=[Const.input_col])
        items = [
            item[Const.input_col] + state.get(key)[Const.input_col].astype(str)
            for key in self.keys
        ]

        state.set(self.name, item)
        return item
