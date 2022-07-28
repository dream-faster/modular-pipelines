from typing import List

from autocorrect import Speller
from type import DataType

from .base import Augmenter


class SpellAutocorrectAugmenter(Augmenter):

    inputTypes = [DataType.List, DataType.Series]
    outputType = DataType.List

    def __init__(self, fast: bool):
        super().__init__()
        self.fast = fast

    def load(self, pipeline_id: str, execution_order: int) -> int:
        self.pipeline_id = pipeline_id
        self.id += f"-{str(execution_order)}"

        self.spell = Speller(fast=self.fast)

        return execution_order + 1

    def predict(self, dataset: List[str]) -> List[str]:
        return [self.spell(text) for text in dataset]
