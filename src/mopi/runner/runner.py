import datetime
from copy import deepcopy
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd
from mopi.constants import Const
from mopi.plugins import IntegrityChecker, PipelineAnalyser
from mopi.plugins.base import Plugin
from mopi.type import Experiment
from mopi.runner.utils import (
    overwrite_dataloaders_,
    overwrite_model_configs_,
    append_parent_path_and_id_,
    add_experiment_config_to_blocks_,
    add_split_category_to_datasource_,
)
from .evaluation import evaluate
from .store import Store
from mopi.utils.printing import logger

obligatory_plugins_begin = []
obligatory_plugins_end = [PipelineAnalyser(), IntegrityChecker()]


class Runner:
    def __init__(
        self,
        experiment: Experiment,
        plugins: List[Optional[Plugin]],
    ) -> None:

        self.experiment = experiment
        self.pipeline = deepcopy(experiment.pipeline)

        add_experiment_config_to_blocks_(self.pipeline, experiment)
        if self.experiment.global_dataloader is not None:
            overwrite_dataloaders_(self.pipeline, experiment.global_dataloader)
        overwrite_model_configs_(experiment, self.pipeline)
        append_parent_path_and_id_(self.pipeline, mask=True)

        self.run_path = f"{Const.output_runs_path}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"
        self.store = Store(dict(), self.run_path)
        add_split_category_to_datasource_(self.pipeline, experiment)
        self.plugins = obligatory_plugins_begin + plugins + obligatory_plugins_end

    def infer(self) -> Tuple[Store, "Pipeline"]:

        logger.log("💈 Loading existing models")
        self.pipeline.load(self.plugins)

        logger.log("🔮 Predicting with pipeline")
        preds_probs = self.pipeline.predict(self.store, self.plugins)
        self.store.set_data(Const.final_output, preds_probs)

        return self.store, self.pipeline

    def train_test(self) -> Tuple[Store, "Pipeline"]:

        logger.log(
            f"Running Experiment in {logger.formats.BOLD}{'TRAINING' if self.experiment.train else 'INFERENCE'}{logger.formats.END} mode"
            + f"\n{logger.formats.CYAN}{self.experiment.project_name} ~ {self.experiment.run_name} {logger.formats.END}",
            mode=logger.modes.BOX,
        )

        for plugin in self.plugins:
            plugin.print_me("on_run_begin")
            self.pipeline = plugin.on_run_begin(self.pipeline)

        logger.log("💈 Loading existing models")
        self.pipeline.load(self.plugins)

        if self.experiment.train:
            logger.log("🏋️ Training pipeline")
            self.pipeline.fit(self.store, self.plugins)

            logger.log("📡 Uploading models")
            self.pipeline.save_remote()

        logger.log("🔮 Predicting with pipeline")
        preds_probs = self.pipeline.predict(self.store, self.plugins)
        self.store.set_data(Const.final_output, preds_probs)

        logger.log("🤔 Evaluating entire pipeline")
        stats = evaluate(
            preds_probs,
            self.pipeline.datasource.get_labels(),
            self.experiment.metrics,
            self.run_path,
        )
        self.store.set_stats(Const.final_eval_name, stats)

        for plugin in self.plugins:
            plugin.print_me("on_run_end")
            _, _ = plugin.on_run_end(self.pipeline, self.store)

        return self.store, self.pipeline
