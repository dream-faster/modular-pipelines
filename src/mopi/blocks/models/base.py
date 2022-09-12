from typing import List, Optional

from mopi.blocks.base import Block, Element
from mopi.type import BaseConfig, Evaluators, Hierarchy


class Model(Block):

    config: BaseConfig
    evaluators: Optional[Evaluators]

    def children(self) -> List[Element]:
        return [self]

    def get_hierarchy(self) -> Hierarchy:
        return Hierarchy(name=self.id, obj=self)
