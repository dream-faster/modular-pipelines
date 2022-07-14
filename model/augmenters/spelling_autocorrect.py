from model.base import Model
from typing import Optional
import pandas as pd
import spacy
from type import BaseConfig
from autocorrect import Speller
from configs import Const
from utils.random import random_string


class SpellAutocorrectAugmenter(Model):
    def __init__(self, fast: bool):
        self.config = BaseConfig(force_fit=False)
        self.fast = fast
        self.id = "spellcorrector" + random_string(5)

    def preload(self):
        self.spell = Speller(fast=self.fast)

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset[Const.input_col] = dataset[Const.input_col].apply(self.spell)
        return dataset

    def is_fitted(self) -> bool:
        return True
