import os
from typing import List, Optional

import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from blocks.models.base import Model
from configs.constants import Const
from datasets import ClassLabel, Features, Value
from datasets.arrow_dataset import Dataset
from sklearn.model_selection import train_test_split
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import enable_full_determinism
from type import DataType, Evaluators, HuggingfaceConfig, LoadOrigin, PredsWithProbs
from utils.env_interface import get_env

from .infer import run_inference
from .train import run_training
from utils.printing import logger
from .loading import safe_load, determine_load_order, get_paths
import time

device = 0 if torch.cuda.is_available() else -1


def initalize_environment_(config: HuggingfaceConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    config.training_args.hub_token = get_env("HF_HUB_TOKEN")
    config.training_args.push_to_hub = config.save_remote


class HuggingfaceModel(Model):

    model: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
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
        self.trainer = None
        self.pretrained = False
        self.evaluators = evaluators

    def load(self) -> None:
        enable_full_determinism(Const.seed)

        paths = get_paths(self.config, self.parent_path, self.id)
        load_order = determine_load_order(self.config, paths)

        for load_origin, load_path in load_order:
            logger.log(
                f"ℹ️ Loading from {logger.formats.BOLD}{load_origin}{logger.formats.END}",
                level=logger.levels.ONE,
            )
            model, tokenizer = safe_load(load_path, config=self.config)

            if model:
                self.model = model
                self.tokenizer = tokenizer

                if load_origin == LoadOrigin.pretrained:
                    self.pretrained = True
                break

        if model is None and self.config.pretrained_model is not None:
            sleep_period = 2.0
            logger.log(
                f"ℹ️ No Model found at first try, sleeping for {sleep_period} and retrying {logger.formats.BOLD}PRETRAINED: {self.config.pretrained_model}{logger.formats.END}"
            )
            time.sleep(sleep_period)
            model, tokenizer = safe_load(
                self.config.pretrained_model, config=self.config
            )
            if model:
                self.model = model
                self.tokenizer = tokenizer
                self.pretrained = True

        self.config.training_args.output_dir = (
            f"{Const.output_pipelines_path}/{self.parent_path}/{self.id}"
        )
        initalize_environment_(self.config)

    def fit(self, dataset: List[str], labels: Optional[pd.Series]) -> None:
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Either a trained model or a pretrained model must be loaded."

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

        self.model = trainer.model
        self.tokenizer = trainer.tokenizer
        self.trainer = trainer

    def predict(self, dataset: pd.Series) -> List[PredsWithProbs]:
        assert not (
            self.pretrained is True and self.is_fitted() is False
        ), "Huggingface model will train during inference as a default if model is not trained! This introduces data leakage."
        assert (
            self.model is not None and self.tokenizer is not None
        ), "Model must be loaded."

        return run_inference(
            model=self.model,
            test_data=from_pandas(
                pd.DataFrame({Const.input_col: dataset}), self.config.num_classes
            ),
            tokenizer=self.tokenizer,
            config=self.config,
            device=device,
        )

    def is_fitted(self) -> bool:
        return self.trainer is not None

    def save(self) -> None:
        if self.model is None or self.tokenizer is None:
            logger.log("Model is not available to save", level=logger.levels.THREE)
        self.model.save_pretrained(
            f"{Const.output_pipelines_path}/{self.parent_path}/{self.id}"
        )
        self.tokenizer.save_pretrained(
            f"{Const.output_pipelines_path}/{self.parent_path}/{self.id}"
        )

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
