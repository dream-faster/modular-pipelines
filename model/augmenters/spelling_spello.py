from model.base import Model
from typing import Optional
import pandas as pd
import spacy
from type import BaseConfig
from spello.model import SpellCorrectionModel
from configs import Const
from utils.random import random_string


class SpellingSpelloAugmenter(Model):
    def __init__(self):
        self.config = BaseConfig(force_fit=False)
        self.id = "spelling-spello" + random_string(5)

    def preload(self):
        self.sp = SpellCorrectionModel(language="en")

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset[Const.input_col] = dataset[Const.input_col].apply(self.sp)
        return dataset

    def is_fitted(self) -> bool:
        return True