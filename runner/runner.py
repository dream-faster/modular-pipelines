import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
from blocks.pipeline import Pipeline
from configs import Const
from configs.constants import LogConst
from plugins import IntegrityChecker, PipelineAnalyser
from plugins.base import Plugin
from type import Evaluators, RunConfig
from utils.flatten import flatten

from .evaluation import evaluate
from .store import Store

obligatory_plugins = [PipelineAnalyser(), IntegrityChecker()]


def print_checker(function_origin, text):
    type(self).on_run_begin.__qualname__.split(".")[0]


def overwrite_model_configs(config: RunConfig, pipeline: Pipeline) -> Pipeline:
    for key, value in vars(config).items():
        if value is not None:
            for model in flatten(pipeline.children()):
                if hasattr(model, "config"):
                    if hasattr(model.config, key):
                        vars(model.config)[key] = value

    return pipeline


class Runner:
    def __init__(
        self,
        run_config: RunConfig,
        pipeline: Pipeline,
        data: Dict[str, Union[pd.Series, List]],
        labels: pd.Series,
        evaluators: Evaluators,
        plugins: List[Optional[Plugin]],
    ) -> None:
        self.config = run_config
        self.run_path = f"{Const.output_runs_path}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"
        self.pipeline = pipeline
        self.store = Store(data, labels, self.run_path)
        self.evaluators = evaluators
        self.plugins = obligatory_plugins + plugins

        self.pipeline = overwrite_model_configs(self.config, self.pipeline)

    def run(self):
        for plugin in self.plugins:
            plugin.print_me("on_run_begin")
            self.pipeline = plugin.on_run_begin(self.pipeline)

        print("💈 Loading existing models")
        self.pipeline.load(self.plugins)

        if self.config.train:
            print("🏋️ Training pipeline")
            self.pipeline.fit(self.store, self.plugins)

            print("📡 Uploading models")
            self.pipeline.save_remote()

        print("🔮 Predicting with pipeline")
        preds_probs = self.pipeline.predict(self.store, self.plugins)
        predictions = [pred[0] for pred in preds_probs]

        print("🤔 Evaluating entire pipeline")
        stats = evaluate(predictions, self.store, self.evaluators, self.run_path)
        self.store.set_stats(Const.final_eval_name, stats)

        for plugin in self.plugins:
            plugin.print_me("on_run_end")
            _, _ = plugin.on_run_end(self.pipeline, self.store)
