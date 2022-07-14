import nltk
from nltk.corpus import wordnet as wn
from .base import Augmenter
from typing import List, Any, Optional
import pandas as pd
from type import BaseConfig
from configs.constants import Const


class SynonymAugmenter(Augmenter):
    def __init__(self, num_synonyms: int):
        super().__init__()
        self.num_synonyms = num_synonyms

    def preload(self):
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:

        dataset[Const.input_col] = dataset[Const.input_col].apply(
            lambda x: " ".join([process_token(token, self.num_synonyms) for token in x])
        )

        return dataset

    def is_fitted(self) -> bool:
        return True


def process_token(token: Any, num_synonyms: int) -> str:
    if token.pos_ == "ADJ" or token.pos_ == "NOUN":
        synonyms = get_synonyms(token.text, num_synonyms)
        if len(synonyms) == 0:
            return token.text
        else:
            return token.text + " (" + " ".join(synonyms) + ")"
    else:
        return token.text


def get_synonyms(word: str, num_synonyms: int) -> List[str]:
    synonyms = set(
        [single_word for ss in wn.synsets(word) for single_word in ss.lemma_names()]
    )
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)[:num_synonyms]
