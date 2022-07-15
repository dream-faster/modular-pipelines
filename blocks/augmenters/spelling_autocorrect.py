from typing import List, Optional
import pandas as pd
from autocorrect import Speller
from .base import Augmenter
from type import DataType


class SpellAutocorrectAugmenter(Augmenter):

    inputTypes = DataType.List_str
    outputType = DataType.List_str

    def __init__(self, fast: bool):
        super().__init__()
        self.fast = fast

    def load_remote(self):
        self.spell = Speller(fast=self.fast)


    def predict(self, dataset: List[str]) -> List[str]:
        return [self.spell(text) for text in dataset]
