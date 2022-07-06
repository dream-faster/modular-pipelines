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

        items = [state.get(key) for key in self.keys]

        final = items[0]
        for item in items[1:]:
            final[Const.input_col] = final[Const.input_col].astype(str) + item[
                Const.input_col
            ].astype(str)

        state.set(self.name, final)
        return final
