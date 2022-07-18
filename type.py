from dataclasses import dataclass
from enum import Enum
from sklearn.base import ClassifierMixin
from typing import List

Label = int
Probabilities = List[float]


class DataType(Enum):
    Any = "Any"
    NpArray = "ndarray"
    List = "List"
    PredictionsWithProbs = "PredictionsWithProbs"
    Series = "Series"
    Tensor = "Tensor"


""" Model Configs """


@dataclass
class BaseConfig:
    force_fit: bool
    save: bool
    save_remote: bool


@dataclass
class HuggingfaceConfig(BaseConfig):
    pretrained_model: str
    epochs: int
    user_name: str
    repo_name: str
    num_classes: int
    val_size: float


@dataclass
class SKLearnConfig(BaseConfig):
    classifier: ClassifierMixin
    one_vs_rest: bool


""" Preprocessing Configs """


@dataclass
class PreprocessConfig:
    train_size: int
    val_size: int
    test_size: int
    input_col: str
    label_col: str


@dataclass
class PytorchConfig(BaseConfig):
    hidden_size: int
    output_size: int
    val_size: float
