from dataclasses import dataclass

from transformers import TrainerCallback
from blocks.models.huggingface.base import HuggingfaceModel
from blocks.pipeline import Pipeline

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


class WandbCallback(TrainerCallback):
    def __init__(self, wandb):
        self.wandb = wandb

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.wandb.log(logs)

    def on_evaluate(self, args, state, control, logs=None, metrics=None, **kwargs):
        self.wandb.log(metrics)


class WandbPlugin(Plugin):
    def __init__(self, config: WandbConfig, configs: Optional[Dict[str, Dict]]):
        super().__init__()
        self.wandb = launch_wandb(config.project_id, configs)

    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        for element in flatten(pipeline.children()):
            if isinstance(element, HuggingfaceModel):
                element.trainer_callbacks = [WandbCallback(wandb=self.wandb)]

        return pipeline

    def on_predict_end(self, store: Store, last_output: Any):
        report_results(output_stats=store.get_all_stats(), wandb=self.wandb, final=True)

        return store, last_output

    def on_run_end(self, pipeline: Pipeline, stats: pd.Series):
        run = wandb.run

        run.save()
        run.finish()


def launch_wandb(
    project_name: str, configs: Optional[Dict[str, Dict]] = None
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
        wandb.init(project=project_name, config=configs, reinit=True)
        return wandb
    except Exception as e:
        logger.debug(e, exc_info=True)


def send_report_to_wandb(stats: pd.Series, wandb: wandb, final: bool = False):
    if wandb is None:
        return

    run = wandb.run
    run.log({"stats": wandb.Table(dataframe=stats)})


def report_results(output_stats: Any, wandb, final: bool = False):
    send_report_to_wandb(output_stats, wandb, final)
