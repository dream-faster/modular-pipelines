from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import TrainerCallback

import wandb
from mopi.blocks.base import DataSource
from mopi.blocks.models.huggingface.base import HuggingfaceModel
from mopi.blocks.pipeline import Pipeline
from mopi.constants import Const
from mopi.runner.store import Store
from mopi.type import BaseConfig, SourceTypes
from mopi.utils.env_interface import get_env
from mopi.utils.list import flatten

from .base import Plugin
from .utils import (
    print_output_statistics,
    print_example_outputs,
    print_correlation_matrix,
    get_output_statistics,
)


@dataclass
class WandbConfig:
    project_id: str
    run_name: str
    train: bool
    delete_run: bool
    output_stats: bool


class WandbCallback(TrainerCallback):
    def __init__(self, wandb, config: WandbConfig):
        self.wandb = wandb
        self.run_type = "train" if config.train else "test"

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.wandb.log({self.run_type: logs})

    def on_evaluate(self, args, state, control, logs=None, metrics=None, **kwargs):
        self.wandb.log({self.run_type: metrics})


class WandbPlugin(Plugin):
    def __init__(
        self, config: WandbConfig, run_config: Optional[Dict[str, Dict]] = None
    ):
        super().__init__()
        self.config = config
        self.run_config = run_config

        self.analysis_functions = [
            ("Output Statistics", print_output_statistics),
            ("Example Outputs", print_example_outputs),
            ("Correlation Matrix", print_correlation_matrix),
        ]

    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        pipeline_config, hierarchy = pipeline.get_configs()
        all_configs = [
            ("run_config", self.run_config),
            ("pipeline_config", pipeline_config),
        ]
        self.wandb = launch_wandb(
            self.config.project_id,
            self.config.run_name + ("_train" if self.config.train is True else "_test"),
            {
                config_name: config
                for config_name, config in all_configs
                if config is not None
            },
            notes="\n\n".join(
                [
                    hierarchy[source_type.value]
                    for source_type in pipeline.get_datasource_types()
                ]
            ),
        )

        for element in flatten(pipeline.children(SourceTypes.fit)):
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
        if self.config.output_stats:
            all_datasources = [
                block
                for block in flatten(pipeline.children(SourceTypes.predict))
                if type(block) is DataSource
            ]

            result_dfs = [
                (
                    datasource,
                    get_output_statistics(
                        store, datasource, self.analysis_functions, log_it=False
                    ),
                )
                for datasource in all_datasources
            ]
            for datasource, output_statistics in result_dfs:
                for name, df in output_statistics:
                    table = wandb.Table(dataframe=df)
                    self.wandb.run.log(
                        {datasource.original_id + "-" + name.replace(" ", "-"): table}
                    )

        if self.wandb is not None:
            run = self.wandb.run

            if run is not None:
                run.save()
                run.finish()

                # if self.config.delete_run:
                # run.delete()

        return pipeline, store


def launch_wandb(
    project_name: str,
    run_name: str,
    configs: Optional[Dict[str, Dict]] = None,
    notes: Optional[str] = None,
) -> Optional[object]:

    wsb_token = get_env("WANDB_API_KEY")

    # try:
    wandb.login(key=wsb_token)
    wandb.init(
        project=project_name,
        config=configs,
        reinit=True,
        name=run_name,
        notes=notes,
    )
    return wandb
    # except Exception as e:
    #     logger.debug(e, exc_info=True)


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
