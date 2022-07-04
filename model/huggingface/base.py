from model.base import BaseModel
from .infer import run_inference_pipeline
from .train import run_training_pipeline
from config import HuggingfaceConfig, huggingface_config
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel
from typing import List, Tuple
from type import Label, Probabilities

class HuggingfaceModel(BaseModel):

    config: HuggingfaceConfig

    def __init__(self, config: HuggingfaceConfig):
        self.config = config

    def fit(self, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame) -> None:
        run_training_pipeline(
            from_pandas(train_dataset), from_pandas(val_dataset), self.config
        )

    def predict(self, test_dataset: pd.DataFrame) -> List[Tuple[Label, Probabilities]]:
        return run_inference_pipeline(from_pandas(test_dataset), huggingface_config)


def from_pandas(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        df,
        features=Features({"text": Value("string"), "label": ClassLabel(5)}),
    )
