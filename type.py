from dataclasses import dataclass
from sklearn.base import ClassifierMixin
from typing import List

Label = int
Probabilities = List[float]

""" Model Configs """


@dataclass
class BaseConfig:
    pass


@dataclass
class HuggingfaceConfig(BaseConfig):
    epochs: int
    user_name: str
    repo_name: str
    push_to_hub: bool = False


@dataclass
class SKLearnConfig(BaseConfig):
    classifier: ClassifierMixin
    one_vs_rest: bool


""" Preprocessing Configs """


@dataclass
class GlobalPreprocessConfig:
    train_size: int
    val_size: int
    test_size: int
    data_from_huggingface: bool
