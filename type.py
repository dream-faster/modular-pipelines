from dataclasses import dataclass
from sklearn.base import ClassifierMixin
from typing import List

Label = int
Probabilities = List[float]

""" Model Configs """


@dataclass
class BaseConfig:
    force_fit: bool


@dataclass
class HuggingfaceConfig(BaseConfig):
    pretrained_model: str
    epochs: int
    user_name: str
    repo_name: str
    num_classes: int
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
