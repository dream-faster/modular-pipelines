from blocks.models.base import Model
from .infer import run_inference_pipeline
from .train import run_training_pipeline
from type import HuggingfaceConfig
from configs.constants import Const
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel
from typing import List, Tuple, Callable, Optional, Union
from transformers import pipeline, Trainer, PreTrainedModel
from sklearn.model_selection import train_test_split

from configs.constants import Const


def safe_load_pipeline(module: Union[str, PreTrainedModel]) -> Optional[Callable]:
    try:
        loaded_pipeline = pipeline(task="sentiment-analysis", model=module)
    except:
        print("Couldn't load pipeline. Skipping.")
        loaded_pipeline = None

    return loaded_pipeline


class HuggingfaceModel(Model):

    config: HuggingfaceConfig

    def __init__(self, id: str, config: HuggingfaceConfig):
        self.id = id
        self.config = config
        self.model: Optional[Union[Callable, Trainer]] = None

    def load(self):
        model = safe_load_pipeline(f"{Const.output_path}/{self.id}")
        if model:
            self.model = model
        else:
            print("     |- ℹ️ No local model found")

    def load_remote(self):
        if self.model is None:
            model = safe_load_pipeline(f"{self.config.user_name}/{self.id}")
            if model:
                self.model = model
            else:
                print(
                    f"     |- ℹ️ No fitted model found remotely, loading pretrained foundational model: {self.config.pretrained_model}"
                )
                self.model = safe_load_pipeline(self.config.pretrained_model)

    def fit(self, dataset: List[str], labels: Optional[pd.Series]) -> None:
        train_dataset, val_dataset = train_test_split(
            pd.DataFrame({Const.input_col: dataset, Const.label_col: labels}),
            test_size=self.config.val_size,
        )

        trainer = run_training_pipeline(
            from_pandas(train_dataset, self.config.num_classes),
            from_pandas(val_dataset, self.config.num_classes),
            self.config,
        )
        self.model = safe_load_pipeline(trainer.model)
        self.trainer = trainer
        self.trained = True

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return run_inference_pipeline(
            self.model,
            from_pandas(dataset, self.config.num_classes),
            self.huggingface_config,
        )

    def is_fitted(self) -> bool:
        return self.model is not None

    def save(self) -> None:
        if self.config.save and self.trained:
            path = f"{Const.output_path}/{self.pipeline_id}/{self.id}"
            self.model.save_pretrained(save_directory=path)

    def save_remote(self) -> None:
        if (self.config.save_remote is not None) and self.trained:
            self.trainer.push_to_hub()


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
