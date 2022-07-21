from dataclasses import dataclass

from transformers import TrainerCallback

from type import BaseConfig
from .base import Plugin
import wandb
from typing import List, Optional, Any, Dict, Tuple
import os
from runner.store import Store
import pandas as pd


@dataclass
class WandbConfig:
    project_id: str


class WandbCallback(TrainerCallback):
    def __init__(self, wandb):
        self.wandb = wandb

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.wandb.log(logs)


class WandbPlugin(Plugin):
    def __init__(self, config: WandbConfig, configs: Optional[Dict[str, Dict]]):
        super().__init__()
        self.wandb = launch_wandb(config.project_id, configs)

        self.trainer_callback = WandbCallback(wandb=self.wandb)

    def on_fit_begin(
        self, store: Store, last_output: Any
    ) -> Tuple[Store, Any, BaseConfig]:
        super().on_fit_begin(store, last_output)

        return store, last_output, [self.trainer_callback]

    def on_predict_end(self, store: Store, last_output: Any):
        super().on_predict_end(store, last_output)
        report_results(output_stats=store.get_all_stats(), wandb=self.wandb, final=True)

        return store, last_output


def get_wandb():
    from dotenv import load_dotenv

    load_dotenv()

    """ 0. Login to Weights and Biases """
    wsb_token = os.environ.get("WANDB_API_KEY")
    if wsb_token:
        wandb.login(key=wsb_token)
        return wandb
    else:
        return None  # wandb.login()


def launch_wandb(
    project_name: str, configs: Optional[Dict[str, Dict]] = None
) -> Optional[object]:
    wandb = get_wandb()
    if wandb is None:
        raise Exception(
            "Wandb can not be initalized, the environment variable WANDB_API_KEY is missing (can also use .env file)"
        )
    else:
        wandb.init(project=project_name, config=configs, reinit=True)
        return wandb


def send_report_to_wandb(
    stats: pd.Series, wandb: Optional[object], final: bool = False
):
    if wandb is None:
        return

    run = wandb.run
    if final:
        run.save()

    run.log({"stats": wandb.Table(dataframe=stats)})

    if final:
        run.finish()


def report_results(output_stats: Any, wandb, final: bool = False):
    send_report_to_wandb(output_stats, wandb, final)
