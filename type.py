from dataclasses import dataclass
from enum import Enum
from sklearn.base import ClassifierMixin
from typing import Callable, List, Optional, Tuple

from transformers import TrainingArguments
from datasets.arrow_dataset import Dataset


Label = int
Probabilities = List[float]

PredsWithProbs = Tuple[Label, Probabilities]


class DataType(Enum):
    Any = "Any"
    NpArray = "ndarray"
    List = "List"
    PredictionsWithProbs = "PredictionsWithProbs"
    Series = "Series"
    Tensor = "Tensor"


EvaluatorId = str
Evaluator = Tuple[EvaluatorId, Callable]
Evaluators = List[Evaluator]


""" Model Configs """


@dataclass
class BaseConfig:
    force_fit: bool
    save: bool
    save_remote: bool


@dataclass
class HuggingfaceConfig(BaseConfig):
    pretrained_model: str
    user_name: str
    repo_name: str
    num_classes: int
    val_size: float
    training_args: TrainingArguments


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


@dataclass
class RunConfig:
    run_name: str  # Get's appended as a prefix before the pipeline name
    train: bool  # Weather the run should do training
    dataset: Dataset
    force_fit: Optional[bool] = None  # If set to True will make all models train
    save_remote: Optional[
        bool
    ] = None  # If set True all models will try uploading (if configured), if set False it overwrites uploading of any models (even if configured)
    remote_logging: Optional[
        bool
    ] = None  # Switches on and off all remote logging (eg.: wandb)
