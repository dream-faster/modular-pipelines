from typing import Any, List

import pandas as pd
import spacy

from constants import Const
from type import DataType
from utils.spacy import get_spacy

from .base import Transformation


class NoLemmatizer(Transformation):

    inputTypes = DataType.List
    outputType = DataType.List

    def __init__(self, remove_stopwords: bool) -> None:
        super().__init__()
        self.remove_stopwords = remove_stopwords

    def load(self) -> None:
        self.nlp = get_spacy()

    def predict(self, dataset: List) -> List[str]:
        return [preprocess(item, self.remove_stopwords) for item in dataset]


def preprocess(tokens: List[Any], remove_stopwords: bool) -> str:
    return " ".join(
        [
            token.text
            for token in tokens
            if (not token.is_stop if remove_stopwords else True)
            and not token.is_punct
            and token.lemma_ != " "
        ]
    )
