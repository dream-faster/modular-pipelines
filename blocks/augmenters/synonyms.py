import nltk
from nltk.corpus import wordnet as wn
from .base import Augmenter
from typing import List, Any, Optional
import pandas as pd
from type import DataType


class SynonymAugmenter(Augmenter):

    inputTypes = DataType.List_str
    outputType = DataType.List_str

    def __init__(self, num_synonyms: int):
        super().__init__()
        self.num_synonyms = num_synonyms

    def load_remote(self):
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    def predict(self, dataset: List) -> List[str]:
        return [
            " ".join([process_token(token, self.num_synonyms) for token in item])
            for item in dataset
        ]


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
