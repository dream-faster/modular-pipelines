import nltk
from nltk.corpus import wordnet as wn
from typing import List, Any, Union, Tuple, Optional
from type import DataType
from configs.constants import Const
from collections import Counter
from urlextract import URLExtract
from .base import Augmenter
from emoji import UNICODE_EMOJI_ENGLISH
import re
import numpy as np


class StatisticAugmenter(Augmenter):

    inputTypes = DataType.List
    outputType = DataType.List

    def predict(self, dataset: List) -> List:
        return [get_statistic([token.text for token in item]) for item in dataset]


def get_num_words(words: list[str]) -> int:
    return len(words)


def get_word_freq(words: list[str]) -> dict:
    return dict(Counter(words))


def get_num_outliers(word_freq: dict) -> int:
    outliers = {}
    ratio = 0.2

    if len(word_freq.keys()) < 10:
        # If there is not enough words return 0
        return 0

    for word, freq in word_freq.items():
        if freq >= (len(word_freq.keys()) * ratio):
            outliers[word] = freq

    return len(outliers.keys())


def get_num_of_urls(string: str) -> int:
    extractor = URLExtract()
    urls = extractor.find_urls(string)
    return len(urls)


def get_non_alphanumeric(string: str) -> int:
    return len([char for char in string if not char.isalnum()])


pattern_same_punctuation = re.compile("(([-/\\\\()!\"+,&'])\\2+)")
pattern_any_punctuation = re.compile("([-/\\\\!?']{2,})")


def get_num_aggressive_char(words_fused: str) -> int:
    match = pattern_any_punctuation.findall(words_fused)
    return len(match) if match is not None else 0


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


def get_distribution_metrics(word_freq: dict) -> Tuple[int, int]:

    distribution = [value for value in word_freq.values()]

    mean = np.mean(distribution)
    variance = np.var(distribution)

    return mean, variance


def get_statistic(words: list[str]) -> List[int]:
    words_fused = " ".join(words)
    num_words = get_num_words(words)
    word_freq = get_word_freq(words)
    mean, variance = get_distribution_metrics(word_freq)
    num_outliers = get_num_outliers(word_freq)
    num_urls = get_num_of_urls(words_fused)
    num_non_alphanumeric = get_non_alphanumeric(words_fused)
    num_uppercase = get_num_uppercase(words)
    num_emoji = get_num_emoji(words_fused)
    num_aggressive_char = get_num_aggressive_char(words_fused)

    return [
        num_words,
        mean,
        variance,
        num_outliers,
        num_urls,
        num_non_alphanumeric,
        num_uppercase,
        num_emoji,
        num_aggressive_char,
    ]
