from dataclasses import dataclass

from transformers import TrainerCallback
from blocks.models.huggingface.base import HuggingfaceModel
from blocks.pipeline import Pipeline
from configs.constants import Const

from type import BaseConfig
from .base import Plugin
import wandb
from typing import List, Optional, Any, Dict, Tuple
import os
from runner.store import Store
import pandas as pd
from utils.flatten import flatten
import logging

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
        self.wandb = launch_wandb(
            config.project_id,
            config.run_name + ("_train" if config.train is True else "_test"),
            configs,
        )
        self.config = config

    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
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
    try:
        from dotenv import load_dotenv

        load_dotenv()

        wsb_token = os.environ.get("WANDB_API_KEY")
        wandb.login(key=wsb_token)

    except Exception as e:
        logger.debug(e, exc_info=True)
        logger.warning("The environment variable WANDB_API_KEY is missing")

    try:
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
