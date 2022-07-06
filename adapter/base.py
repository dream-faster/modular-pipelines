from data.state import GlobalState
from typing import List, Optional


class BaseAdapter:
    def __init__(
        self, name: Optional[str], keys: Optional[List[str]], key: Optional[str]
    ):
        pass

    def preload(self):
        pass

    def get_parent_class(self):
        for base in self.__class__.__bases__:
            return base

    def connect(self, state: GlobalState):
        raise NotImplementedError()
