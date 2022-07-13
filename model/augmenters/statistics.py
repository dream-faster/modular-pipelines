import nltk
from nltk.corpus import wordnet as wn
from typing import List, Any, Union
import pandas as pd
import spacy
from type import BaseConfig
from configs.constants import Const
from collections import Counter
from urlextract import URLExtract
from .base import BaseAugmenter
from emoji import UNICODE_EMOJI_ENGLISH


class StatisticAugmenter(BaseAugmenter):
    def __init__(self, id: str):
        self.config = BaseConfig(force_fit=False)
        self.id = id

    def preload(self):
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        self.nlp = spacy.load("en_core_web_lg")

    def fit(self, dataset: pd.DataFrame):
        pass

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset[Const.input_col] = dataset[Const.input_col].apply(
            lambda x: get_statistic([token.text for token in self.nlp(x)])
        )
        return dataset

    def is_fitted(self) -> bool:
        return True


def get_num_words(words: list[str]) -> int:
    return len(words)


def get_word_freq(words: list[str]) -> dict:
    return dict(Counter(words))


def get_outliers(word_freq: dict) -> dict:
    outliers = {}
    ratio = 0.1

    for word, freq in word_freq.items():
        if freq >= (len(word_freq.keys()) * ratio):
            outliers[word] = freq

    return outliers


def get_num_of_urls(string: str) -> int:
    extractor = URLExtract()
    urls = extractor.find_urls(string)
    return len(urls)


def get_non_alphanumeric(string: str) -> int:
    return len([char for char in string if not char.isalnum()])


def get_num_emoji(words_fused: str) -> int:
    return len([char for char in words_fused if char in UNICODE_EMOJI_ENGLISH])


def get_num_uppercase(words: list[str]) -> int:
    return len(
        [
            word
            for word in words
            if all([char for char in word if char.isupper()]) is True
        ]
    )


def get_statistic(words: list[str]) -> List[Union[int, dict]]:
    words_fused = " ".join(words)
    num_words = get_num_words(words)
    word_freq = get_word_freq(words)
    outliers = get_outliers(word_freq)
    num_urls = get_num_of_urls(words_fused)
    num_non_alphanumeric = get_non_alphanumeric(words_fused)
    num_uppercase = get_num_uppercase(words)
    num_emoji = get_num_emoji(words_fused)

    return [
        num_words,
        word_freq,
        outliers,
        num_urls,
        num_non_alphanumeric,
        num_uppercase,
        num_emoji,
    ]
