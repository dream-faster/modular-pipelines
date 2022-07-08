import nltk
from nltk.corpus import wordnet as wn
from model.base import BaseModel
from typing import List, Any
import pandas as pd
import spacy
from type import BaseConfig


class SynonymAugmenter(BaseModel):
    def __init__(self, num_synonyms: int):
        self.config = BaseConfig(force_fit=False)
        self.num_synonyms = num_synonyms

    def preload(self):
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        self.nlp = spacy.load("en_core_web_lg")

    def fit(self, train_dataset: pd.DataFrame):
        pass

    def predict(self, test_dataset: pd.DataFrame) -> List[str]:
        return [
            " ".join(
                [process_token(token, self.num_synonyms) for token in self.nlp(line)]
            )
            for line in test_dataset["input"]
        ]

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
