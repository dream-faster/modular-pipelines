from adapter.base import BaseAdapter
from data.state import GlobalState
from typing import List
from configs.constants import Const


class IdentityAdapter(BaseAdapter):
    def __init__(self, key: str):
        self.key = key

    def connect(self, state: GlobalState):
        return state.get(self.key)
