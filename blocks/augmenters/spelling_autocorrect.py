from blocks.models.base import Model
from typing import List, Optional
import pandas as pd
import spacy
from type import BaseConfig
from autocorrect import Speller
from configs import Const
from utils.random import random_string
from .base import Augmenter


class SpellAutocorrectAugmenter(Augmenter):
    def __init__(self, fast: bool):
        super().__init__()
        self.fast = fast

    def preload(self):
        self.spell = Speller(fast=self.fast)

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: List[str]) -> List[str]:
        return [self.spell(text) for text in dataset]

    def is_fitted(self) -> bool:
        return True
