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

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

device = 0 if torch.cuda.is_available() else -1


def multi_line_print(text: str) -> None:
    print(
        textwrap.fill(
            text,
            initial_indent="    ┃  ├── ",
            subsequent_indent="    ┃  │   ",
            width=100,
        )
    )


def safe_load_training_pipeline(
    module: str, config: HuggingfaceConfig
) -> Tuple[
    Optional[AutoModelForSequenceClassification], Optional[PreTrainedTokenizerBase]
]:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            module, num_labels=config.num_classes
        )
        tokenizer = AutoTokenizer.from_pretrained(module)

        multi_line_print(
            f"Training: Model loaded {config.task_type.value}: {PrintFormats.BOLD}{module.__class__.__name__ if isinstance(module, PreTrainedModel) else module}{PrintFormats.END}"
        )
    except:
        multi_line_print(
            f"Training: Couldn't load {module} model or tokenizer. Skipping."
        )
        model, tokenizer = None, None

    return model, tokenizer


def safe_load_inference_pipeline(
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

        multi_line_print(
            f"Inference: Pipeline loaded {config.task_type.value}: {PrintFormats.BOLD}{module.__class__.__name__ if isinstance(module, PreTrainedModel) else module}{PrintFormats.END}"
        )

    except:
        multi_line_print(f"Inference: Couldn't load {module} pipeline. Skipping.")

        loaded_pipeline = None

    return loaded_pipeline


def safe_load(
    train: bool,
    module: Union[str, PreTrainedModel],
    config: HuggingfaceConfig,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Tuple[
    Optional[Union[Callable, PreTrainedModel]], Optional[PreTrainedTokenizerBase]
]:

    if train and isinstance(module, str):
        return safe_load_training_pipeline(module, config)
    else:
        return safe_load_inference_pipeline(module, config, tokenizer)


def initalize_environment_(config: HuggingfaceConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    config.training_args.hub_token = get_env("HF_HUB_TOKEN")


def get_paths(config: HuggingfaceConfig, parent_path: str, id: str) -> dict:
    return {
        LoadOrigin.local: f"{Const.output_pipelines_path}/{parent_path}/{id}",
        LoadOrigin.remote: f"{config.user_name}/{id}"
        if not hasattr(config, "remote_name_override")
        or config.remote_name_override is None
        else config.remote_name_override,
        LoadOrigin.pretrained: config.pretrained_model,
    }


def determine_load_order(
    config: HuggingfaceConfig, paths: dict
) -> List[Tuple[LoadOrigin, str]]:
    if (
        hasattr(config, "preferred_load_origin")
        and config.preferred_load_origin is not None
    ):
        load_order = [
            (
                config.preferred_load_origin,
                paths[config.preferred_load_origin],
            )
        ]
    else:
        load_order = paths.items()

    return load_order


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
        self.model = None
        self.trained = False
        self.pretrained = False
        self.evaluators = evaluators

        initalize_environment_(self.config)

    def load(self) -> None:
        enable_full_determinism(Const.seed)

        paths = get_paths(self.config, self.parent_path, self.id)
        load_order = determine_load_order(self.config, paths)

        for load_origin, load_path in load_order:
            print(
                f"    ┣━━┯ ℹ️ Loading from {PrintFormats.BOLD}{load_origin}{PrintFormats.END}"
            )
            model, tokenizer = safe_load(
                self.run_context.train, load_path, config=self.config
            )

            if model:
                self.model = model

                if tokenizer:
                    self.tokenizer = tokenizer

                if load_origin == LoadOrigin.pretrained:
                    self.pretrained = True
                break

        self.config.training_args.output_dir = (
            f"{Const.output_pipelines_path}/{self.parent_path}/{self.id}"
        )

    def fit(self, dataset: List[str], labels: Optional[pd.Series]) -> None:
        train_dataset, val_dataset = train_test_split(
            pd.DataFrame({Const.input_col: dataset, Const.label_col: labels}),
            test_size=self.config.val_size,
        )

        trainer = run_training(
            self.model,
            self.tokenizer,
            self.config.training_args,
            from_pandas(train_dataset, self.config.num_classes),
            from_pandas(val_dataset, self.config.num_classes),
            self.trainer_callbacks if hasattr(self, "trainer_callbacks") else None,
        )

        self.model = safe_load(
            self.run_context.train, trainer.model, self.config, trainer.tokenizer
        )

        self.trainer = trainer
        self.trained = True

    def predict(self, dataset: pd.Series) -> List[PredsWithProbs]:
        assert not (
            self.pretrained is True and self.trained is False
        ), "Huggingface model will train during inference as a default if model is not trained! This introduces data leakage."

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
