from numpy import True_
from model.base import BaseModel
from .infer import run_inference_pipeline
from .train import run_training_pipeline
from config import HuggingfaceConfig, huggingface_config
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel
from typing import List, Tuple, Callable, Optional, Any
from type import Label, Probabilities
import os.path
from transformers import pipeline


def load_module(module_name: str) -> Callable:
    return pipeline(task="sentiment-analysis", model=module_name)


class HuggingfaceModel(BaseModel):

    config: HuggingfaceConfig

    def __init__(self, config: HuggingfaceConfig):
        self.config: HuggingfaceConfig = config

    def fit(self, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame) -> None:
        if not self.fitted_model:
            run_training_pipeline(
                from_pandas(train_dataset),
                from_pandas(val_dataset),
                self.config,
            )

    def predict(self, test_dataset: pd.DataFrame) -> List[Tuple[Label, Probabilities]]:
        model = self.load_fitted(self.config)
        return run_inference_pipeline(
            model, from_pandas(test_dataset), huggingface_config
        )

    def load_fitted(self, config: HuggingfaceConfig) -> Optional[Any]:
        try:
            model = load_module(config.user_name + "/" + config.repo_name)
            print(
                f'⬇️ Downloading model found on {config.user_name + "/" + config.repo_name}'
            )
        except:
            print(f'0️⃣ No model found on {config.user_name + "/" + config.repo_name}')
            model = None

        return model

    def is_fitted(self):
        if self.load_fitted(self.config) is not None:
            return True
        else:
            return False


def from_pandas(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        df,
        features=Features({"text": Value("string"), "label": ClassLabel(5)}),
    )
