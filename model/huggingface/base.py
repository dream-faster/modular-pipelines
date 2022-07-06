from numpy import True_
from model.base import Block
from .infer import run_inference_pipeline
from .train import run_training_pipeline
from configs.config import HuggingfaceConfig, huggingface_config
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel
from typing import List, Tuple, Callable, Optional, Union, Any
from transformers import pipeline, Trainer, PreTrainedModel
from sklearn.model_selection import train_test_split

from configs.constants import Const


def load_pipeline(module: Union[str, PreTrainedModel]) -> Callable:
    return pipeline(task="sentiment-analysis", model=module)


class HuggingfaceModel(Block):

    config: HuggingfaceConfig

    def __init__(self, config: HuggingfaceConfig):
        self.config: HuggingfaceConfig = config
        self.model: Optional[Union[Callable, Trainer]] = None

    def preload(self):
        repo_name = self.config.user_name + "/" + self.config.repo_name
        try:
            self.model = load_pipeline(repo_name)
        except:
            print("❌ No model found in huggingface repository")

    def fit(self, train_dataset: pd.DataFrame) -> None:

        train_dataset, val_dataset = train_test_split(
            train_dataset, test_size=self.config.val_size
        )

        model = run_training_pipeline(
            from_pandas(train_dataset, self.config.num_classes),
            from_pandas(val_dataset, self.config.num_classes),
            self.config,
        )
        self.model = load_pipeline(model)

    def predict(self, test_dataset: pd.DataFrame) -> pd.DataFrame:
        if self.model:
            model = self.model
        else:
            print(
                f"❌ No fitted model, using inference on pretrained foundational model :{self.config.pretrained_model}"
            )
            model = load_pipeline(self.config.pretrained_model)

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
        features=Features(
            {
                Const.input_col: Value("string"),
                Const.label_col: ClassLabel(num_classes),
            }
        ),
    )
