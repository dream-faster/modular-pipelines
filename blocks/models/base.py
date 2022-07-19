from type import BaseConfig, Evaluators
from blocks.base import Block, Element
from typing import List, Optional


class Model(Block):

    config: BaseConfig
    evaluators: Optional[Evaluators]

    def children(self) -> List[Element]:
        return [self]
