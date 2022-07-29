from typing import List, Optional

from blocks.base import Block, Element
from type import BaseConfig, Evaluators


class Model(Block):

    config: BaseConfig
    evaluators: Optional[Evaluators]

    def children(self) -> List[Element]:
        return [self]

    def dict_children(self) -> dict:
        return {"name": self.id, "obj": self}
