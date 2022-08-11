import datetime
from copy import deepcopy
from typing import Dict, List, Optional, Union

import pandas as pd
from configs import Const
from plugins import IntegrityChecker, PipelineAnalyser
from plugins.base import Plugin
from type import Experiment
from utils.run_helpers import (
    overwrite_model_configs_,
    append_parent_path_and_id_,
    add_experiment_config_to_blocks_,
)
from .evaluation import evaluate
from .store import Store


obligatory_plugins_begin = []
obligatory_plugins_end = [PipelineAnalyser(), IntegrityChecker()]


class Runner:
    def __init__(
        self,
        experiment: Experiment,
        data: Dict[str, Union[pd.Series, List]],
        labels: pd.Series,
        plugins: List[Optional[Plugin]],
    ) -> None:

        self.experiment = experiment
        self.pipeline = deepcopy(experiment.pipeline)

        add_experiment_config_to_blocks_(self.pipeline, experiment)
        overwrite_model_configs_(experiment, self.pipeline)
        append_parent_path_and_id_(self.pipeline)

        self.run_path = f"{Const.output_runs_path}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"
        self.store = Store(data, labels, self.run_path)
        self.plugins = obligatory_plugins_begin + plugins + obligatory_plugins_end

    def run(self):
        for plugin in self.plugins:
            plugin.print_me("on_run_begin")
            self.pipeline = plugin.on_run_begin(self.pipeline)

        print("💈 Loading existing models")
        self.pipeline.load(self.plugins)

        if self.experiment.train:
            print("🏋️ Training pipeline")
            self.pipeline.fit(self.store, self.plugins)

            print("📡 Uploading models")
            self.pipeline.save_remote()

        print("🔮 Predicting with pipeline")
        preds_probs = self.pipeline.predict(self.store, self.plugins)
        self.store.set_data(Const.final_output, preds_probs)

        print("🤔 Evaluating entire pipeline")
        stats = evaluate(
            preds_probs, self.store, self.experiment.metrics, self.run_path
        )
        self.store.set_stats(Const.final_eval_name, stats)

        for plugin in self.plugins:
            plugin.print_me("on_run_end")
            _, _ = plugin.on_run_end(self.pipeline, self.store)
