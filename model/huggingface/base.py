from numpy import True_
from model.base import BaseModel
from .infer import run_inference_pipeline
from .train import run_training_pipeline
from config import HuggingfaceConfig, huggingface_config
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel
from typing import List, Tuple, Callable, Optional
from type import Label, Probabilities
from transformers import pipeline


def load_module(module_name: str) -> Callable:
    return pipeline(task="sentiment-analysis", model=module_name)


class HuggingfaceModel(BaseModel):

    config: HuggingfaceConfig

    def __init__(self, config: HuggingfaceConfig):
        self.config: HuggingfaceConfig = config
        self.model: Optional[Callable] = None

    def preload(self):
        repo_name = self.config.user_name + "/" + self.config.repo_name
        try:
            self.model = load_module(repo_name)
        except:
            pass

    def fit(self, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame) -> None:
        run_training_pipeline(
            from_pandas(train_dataset, self.config.num_classes),
            from_pandas(val_dataset, self.config.num_classes),
            self.config,
        )

    def predict(self, test_dataset: pd.DataFrame) -> List[Tuple[Label, Probabilities]]:
        if self.model:
            model = self.model
        else:
            print(
                f"❌ No fitted model, using inference on pretrained foundational model :{self.config.pretrained_model}"
            )
            model = load_module(self.config.pretrained_model)

        return run_inference_pipeline(
            model,
            from_pandas(test_dataset, self.config.num_classes),
            huggingface_config,
        )

    def is_fitted(self) -> bool:
        return self.model is not None


def from_pandas(df: pd.DataFrame, num_classes: int) -> Dataset:
    return Dataset.from_pandas(
        df,
        features=Features({"text": Value("string"), "label": ClassLabel(num_classes)}),
    )
