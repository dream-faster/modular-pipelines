from type import BaseConfig
from blocks.base import Block, Element
from typing import List


class Model(Block):

    config: BaseConfig

    def children(self) -> List[Element]:
        return [self]
