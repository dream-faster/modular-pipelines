from dataclasses import dataclass
from enum import Enum
import pandas as pd
from sklearn.base import ClassifierMixin
from typing import Callable, List, Literal, Optional, Tuple, Union
import pandas as pd
from transformers.training_args import TrainingArguments

from configs.constants import Const


TrainDataset = pd.DataFrame
TestDataset = pd.DataFrame

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
    preferred_load_origin: Literal[Const.remote, Const.local, Const.pretrained, None]


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


@dataclass
class PytorchConfig(BaseConfig):
    hidden_size: int
    output_size: int
    val_size: float


""" Preprocessing Configs """


@dataclass
class PreprocessConfig:
    train_size: int
    val_size: int
    test_size: int
    input_col: str
    label_col: str

    def get_configs(self):
        return vars(self)


""" Run Configs """


@dataclass
class RunConfig:
    run_name: str  # Get's appended as a prefix before the pipeline name
    train: bool  # Weather the run should do training
    dataset: pd.DataFrame
    force_fit: Optional[bool] = None  # If set to True will make all models train
    save_remote: Optional[
        bool
    ] = None  # If set True all models will try uploading (if configured), if set False it overwrites uploading of any models (even if configured)
    remote_logging: Optional[
        bool
    ] = None  # Switches on and off all remote logging (eg.: wandb)

    def get_configs(self):
        return vars(self)
