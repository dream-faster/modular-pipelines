from blocks.models.base import Model
from .infer import run_inference_pipeline
from .train import run_training_pipeline
from type import HuggingfaceConfig, DataType
from configs.constants import Const
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel
from typing import List, Tuple, Callable, Optional, Union
from transformers import (
    pipeline,
    Trainer,
    PreTrainedModel,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
)
from sklearn.model_selection import train_test_split

from configs.constants import Const


def safe_load_pipeline(
    module: Union[str, PreTrainedModel],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Optional[Callable]:
    try:
        if tokenizer is not None:
            loaded_pipeline = pipeline(
                task="sentiment-analysis", model=module, tokenizer=tokenizer
            )
        else:
            loaded_pipeline = pipeline(task="sentiment-analysis", model=module)
        print(f"     ├ Pipeline {module} loaded")
    except:
        print(f"     ├ Couldn't load {module} pipeline. Skipping.")
        loaded_pipeline = None

    return loaded_pipeline


class HuggingfaceModel(Model):

    config: HuggingfaceConfig
    inputTypes = [DataType.Series, DataType.List]
    outputType = DataType.PredictionsWithProbs

    def __init__(self, id: str, config: HuggingfaceConfig):
        self.id = id
        self.config = config
        self.model: Optional[Union[Callable, Trainer]] = None
        self.pretrained: bool = False

    def load(self, pipeline_id: str, execution_order: int) -> None:
        self.pipeline_id = pipeline_id
        self.id += f"-{str(execution_order)}"

        model = safe_load_pipeline(f"{Const.output_path}/{self.pipeline_id}/{self.id}")
        if model:
            self.model = model
        else:
            print("     ├ ℹ️ No local model found")

        return execution_order + 1

    def load_remote(self):
        if self.model is None:
            model = safe_load_pipeline(f"{self.config.user_name}/{self.id}")
            if model:
                self.model = model
            else:
                print(
                    f"     ├ ℹ️ No fitted model found remotely, loading pretrained foundational model: {self.config.pretrained_model}"
                )
                self.model = safe_load_pipeline(self.config.pretrained_model)
                self.pretrained = True

    def fit(self, dataset: List[str], labels: Optional[pd.Series]) -> None:
        train_dataset, val_dataset = train_test_split(
            pd.DataFrame({Const.input_col: dataset, Const.label_col: labels}),
            test_size=self.config.val_size,
        )

        trainer = run_training_pipeline(
            from_pandas(train_dataset, self.config.num_classes),
            from_pandas(val_dataset, self.config.num_classes),
            self.config,
            self.pipeline_id,
            self.id,
        )
        self.model = safe_load_pipeline(trainer.model, trainer.tokenizer)

        self.trainer = trainer
        self.trained = True

    def predict(self, dataset: pd.Series) -> pd.DataFrame:
        return run_inference_pipeline(
            self.model,
            from_pandas(
                pd.DataFrame({Const.input_col: dataset}), self.config.num_classes
            ),
            self.config,
        )

    def is_fitted(self) -> bool:
        return self.pretrained is False

    def save(self) -> None:
        pass
        # if self.config.save and self.trained:
        #     path = f"{Const.output_path}/{self.pipeline_id}/{self.id}"
        #     self.model.save_pretrained(save_directory=path)

    def save_remote(self) -> None:
        if (self.config.save_remote is not None) and self.trained:
            self.trainer.push_to_hub()


def from_pandas(df: pd.DataFrame, num_classes: int = None) -> Dataset:

    if Const.label_col in df.columns:
        return Dataset.from_pandas(
            df,
            features=Features(
                {
                    Const.input_col: Value("string"),
                    Const.label_col: ClassLabel(num_classes),
                }
            ),
        )
    else:
        return Dataset.from_pandas(
            df,
            features=Features(
                {
                    Const.input_col: Value("string"),
                }
            ),
        )
