import nltk
from nltk.corpus import wordnet as wn
from model.base import BaseModel
from typing import List
import pandas as pd

from type import BaseConfig


class SynonymAugmenter(BaseModel):
    def __init__(self):
        self.config = BaseConfig(force_fit=False)

    def preload(self):
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    def fit(self, train_dataset: pd.DataFrame):
        pass

    def predict(self, test_dataset: pd.DataFrame) -> List[List[str]]:
        return [
            [single_word for ss in wn.synsets(word) for single_word in ss.lemma_names()]
            for word in test_dataset.iterrows()
        ]

    def is_fitted(self) -> bool:
        return True
