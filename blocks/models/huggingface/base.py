import os
from threading import local
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import torch
from blocks.models.base import Model
from configs.constants import Const
from datasets import ClassLabel, Dataset, Features, Value
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    enable_full_determinism,
    pipeline,
)
from type import DataType, Evaluators, HuggingfaceConfig, LoadOrigin, PredsWithProbs
from utils.env_interface import get_env

from .infer import run_inference
from .train import run_training
import textwrap
from utils.printing import PrintFormats

device = 0 if torch.cuda.is_available() else -1


def safe_load_pipeline(
    module: Union[str, PreTrainedModel],
    config: HuggingfaceConfig,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Optional[Callable]:
    try:
        if tokenizer is not None:
            loaded_pipeline = pipeline(
                task=config.task_type.value,
                model=module,
                tokenizer=tokenizer,
                device=device,
            )
        else:
            loaded_pipeline = pipeline(
                task=config.task_type.value, model=module, device=device
            )

        print(
            textwrap.fill(
                f"Pipeline loaded {config.task_type.value}: {PrintFormats.BOLD}{module.__class__.__name__ if isinstance(module, PreTrainedModel) else module}{PrintFormats.END}",
                initial_indent="    ┃  ├── ",
                subsequent_indent="    ┃  │   ",
                width=100,
            )
        )
    except:
        print(
            textwrap.fill(
                f"Couldn't load {module} pipeline. Skipping.",
                initial_indent="    ┃  ├── ",
                subsequent_indent="    ┃  │   ",
                width=100,
            )
        )
        loaded_pipeline = None

    return loaded_pipeline


class HuggingfaceModel(Model):

    config: HuggingfaceConfig
    inputTypes = [DataType.Series, DataType.List]
    outputType = DataType.PredictionsWithProbs

    def __init__(
        self,
        id: str,
        config: HuggingfaceConfig,
        evaluators: Optional[Evaluators] = None,
    ):
        self.id = id
        self.config = config
        self.model: Optional[Union[Callable, Trainer]] = None
        self.trained = False
        self.evaluators = evaluators

        os.environ["TOKENIZERS_PARALLELISM"] = "False"

        self.training_args = self.config.training_args
        self.training_args.hub_token = get_env("HF_HUB_TOKEN")
        self.pretrained = False

    def load(self) -> None:
        enable_full_determinism(Const.seed)

        paths = {
            LoadOrigin.local: f"{self.parent_path}/{self.id}",
            LoadOrigin.remote: f"{self.config.user_name}/{self.id}"
            if not hasattr(self.config, "remote_name_override")
            or self.config.remote_name_override is None
            else self.config.remote_name_override,
            LoadOrigin.pretrained: self.config.pretrained_model,
        }

        if (
            hasattr(self.config, "preferred_load_origin")
            and self.config.preferred_load_origin is not None
        ):
            load_order = [
                (
                    self.config.preferred_load_origin,
                    paths[self.config.preferred_load_origin],
                )
            ]
        else:
            load_order = paths.items()

        for key, load_path in load_order:
            print(
                f"    ┣━━┯ ℹ️ Loading from {PrintFormats.BOLD}{key}{PrintFormats.END}"
            )
            model = safe_load_pipeline(load_path, config=self.config)
            if model:
                self.model = model

                if key == LoadOrigin.pretrained:
                    self.pretrained = True
                break

        self.training_args.output_dir = f"{self.parent_path}/{self.id}"

    def fit(self, dataset: List[str], labels: Optional[pd.Series]) -> None:

        train_dataset, val_dataset = train_test_split(
            pd.DataFrame({Const.input_col: dataset, Const.label_col: labels}),
            test_size=self.config.val_size,
        )

        trainer = run_training(
            self.training_args,
            from_pandas(train_dataset, self.config.num_classes),
            from_pandas(val_dataset, self.config.num_classes),
            self.config,
            self.parent_path,
            self.id,
            self.trainer_callbacks if hasattr(self, "trainer_callbacks") else None,
        )
        self.model = safe_load_pipeline(trainer.model, self.config, trainer.tokenizer)

        self.trainer = trainer
        self.trained = True

    def predict(self, dataset: pd.Series) -> List[PredsWithProbs]:
        assert not (
            self.pretrained is True and self.trained is False
        ), "Huggingface model will train during inference (test) only! This introduces data leakage."

        return run_inference(
            self.model,
            from_pandas(
                pd.DataFrame({Const.input_col: dataset}), self.config.num_classes
            ),
            self.config,
        )

    def is_fitted(self) -> bool:
        return self.trained

    def save_remote(self) -> None:
        if all([self.config.save_remote, self.config.save]) is True:
            self.trainer.push_to_hub()


def from_pandas(df: pd.DataFrame, num_classes: int) -> Dataset:

    if Const.label_col in df.columns:
        return Dataset.from_pandas(
            df,
            features=Features(
                {
                    Const.input_col: Value("string"),
                    Const.label_col: ClassLabel(num_classes),
                }
            ),
            preserve_index=False,
        )
    else:
        return Dataset.from_pandas(
            df,
            features=Features(
                {
                    Const.input_col: Value("string"),
                }
            ),
            preserve_index=False,
        )
