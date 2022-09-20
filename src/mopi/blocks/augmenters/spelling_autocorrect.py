from typing import List

from autocorrect import Speller

from mopi.type import DataType

from .base import Augmenter


class SpellAutocorrectAugmenter(Augmenter):

    inputTypes = [DataType.List, DataType.Series]
    outputType = DataType.List

    def __init__(self, fast: bool):
        super().__init__()
        self.fast = fast

    def load(self) -> None:
        self.model = Speller(fast=self.fast)

    def predict(self, dataset: List[str]) -> List[str]:
        return [self.model(text) for text in dataset]
