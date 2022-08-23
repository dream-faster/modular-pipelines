import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import TrainerCallback

import wandb
from blocks.models.huggingface.base import HuggingfaceModel
from blocks.pipeline import Pipeline
from constants import Const
from runner.store import Store
from type import BaseConfig
from utils.env_interface import get_env
from utils.list import flatten

from .base import Plugin

logger = logging.getLogger("Wandb-Plugin")


@dataclass
class WandbConfig:
    project_id: str
    run_name: str
    train: bool


class WandbCallback(TrainerCallback):
    def __init__(self, wandb, config: WandbConfig):
        self.wandb = wandb
        self.run_type = "train" if config.train else "test"

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.wandb.log({self.run_type: logs})

    def on_evaluate(self, args, state, control, logs=None, metrics=None, **kwargs):
        self.wandb.log({self.run_type: metrics})


class WandbPlugin(Plugin):
    def __init__(self, config: WandbConfig, configs: Optional[Dict[str, Dict]]):
        super().__init__()
        self.config = config
        self.configs = configs

    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        self.wandb = launch_wandb(
            self.config.project_id,
            self.config.run_name + ("_train" if self.config.train is True else "_test"),
            self.configs,
        )

        for element in flatten(pipeline.children()):
            if isinstance(element, HuggingfaceModel):
                element.trainer_callbacks = [
                    WandbCallback(wandb=self.wandb, config=self.config)
                ]

        return pipeline

    def on_predict_end(self, store: Store, last_output: Any):
        report_results(
            stats=store.get_all_stats(), wandb=self.wandb, config=self.config
        )

        return store, last_output

    def on_run_end(self, pipeline: Pipeline, store: Store):
        report_results(
            stats=store.get_all_stats(), wandb=self.wandb, config=self.config
        )

        if self.wandb is not None:
            run = self.wandb.run

            if run is not None:
                run.save()
                run.finish()

        return pipeline, store


def launch_wandb(
    project_name: str, run_name: str, configs: Optional[Dict[str, Dict]] = None
) -> Optional[object]:

    wsb_token = get_env("WANDB_API_KEY")

    try:
        wandb.login(key=wsb_token)
        wandb.init(project=project_name, config=configs, reinit=True, name=run_name)
        return wandb
    except Exception as e:
        logger.debug(e, exc_info=True)


def report_results(stats: pd.DataFrame, wandb: wandb, config: WandbConfig):
    if wandb is None or len(stats) < 1:
        return

    run = wandb.run

    run_type = "train" if config.train else "test"
    # Log values of the final run seperately
    if Const.final_eval_name in stats.columns:
        for index, value in stats[Const.final_eval_name].items():
            run.log({run_type: {index: value}})

    run.log({run_type: {"stats": wandb.Table(dataframe=stats)}})
