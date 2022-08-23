from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

import pandas as pd
from sklearn.base import ClassifierMixin
from transformers.training_args import TrainingArguments
from utils.setter import Settable

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
Evaluator = Tuple[EvaluatorId, Callable[[List, List[PredsWithProbs]], Any]]
Evaluators = List[Evaluator]


""" Model Configs """


class LoadOrigin(Enum):
    remote = "remote"
    local = "local"
    pretrained = "pretrained"


class HFTaskTypes(Enum):
    sentiment_analysis = "sentiment-analysis"
    text_classification = "text-classification"


@dataclass
class BaseConfig(Settable):
    frozen: bool
    save: bool
    save_remote: bool
    preferred_load_origin: Optional[LoadOrigin]


@dataclass
class HuggingfaceConfig(BaseConfig):
    pretrained_model: str
    task_type: HFTaskTypes
    user_name: str
    num_classes: int
    val_size: float
    training_args: TrainingArguments
    remote_name_override: Optional[str] = None


@dataclass
class SKLearnConfig(BaseConfig):
    classifier: ClassifierMixin
    one_vs_rest: bool  # this is expensive, uses https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
    calibrate: bool  # this is expensive, uses https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html


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


""" Experiment """


class DatasetSplit(Enum):
    train = "train"
    val = "validation"
    test = "test"


@dataclass
class Experiment:
    project_name: str
    run_name: str  # Get's appended as a prefix before the pipeline name
    train: bool  # Weather the run should do training
    dataset_category: DatasetSplit
    pipeline: "Pipeline"
    metrics: Evaluators
    global_dataloader: Optional["DataLoader"] # If set, will override all DataSource's DataLoaders
    force_fit: Optional[bool] = None  # If set to True will make all models train

    def get_configs(self):
        return vars(self)


class StagingNames(Enum):
    dev = "development"
    prod = "production"
    exp = "experiment"


class SourceTypes(Enum):
    fit = "fit"
    predict = "predict"


@dataclass
class StagingConfig:
    name: StagingNames
    save_remote: Optional[
        bool
    ]  # If set True all models will try uploading (if configured), if set False it overwrites uploading of any models (even if configured)
    log_remote: Optional[bool]  # Switches on and off all remote logging (eg.: wandb)
    limit_dataset_to: Optional[int]


@dataclass
class RunContext:
    project_name: str
    run_name: str
    train: bool


@dataclass
class Hierarchy:
    name: str
    obj: "Element"
    children: Optional[List["Hierarchy"]] = None

    def print_hierarchy(self, child=None, indentation: str = "") -> None:
        el = self if child is None else child
        for key, value in vars(el).items():
            if key == "name":
                print(f"{indentation}- {value}")
            if key == "children" and value is not None:
                for child in value:
                    self.print_hierarchy(child, indentation + "    ")
