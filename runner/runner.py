import datetime
from copy import deepcopy
from typing import Dict, List, Optional, Union

import pandas as pd
from blocks.base import Block, DataSource, Element
from blocks.pipeline import Pipeline
from configs import Const
from plugins import IntegrityChecker, PipelineAnalyser
from plugins.base import Plugin
from type import Experiment
from utils.flatten import flatten

from .evaluation import evaluate
from .store import Store

obligatory_plugins = [PipelineAnalyser(), IntegrityChecker()]


def overwrite_model_configs_(config: Experiment, pipeline: Pipeline) -> None:
    """
    Takes global config values and overwrites the config of
    each model in the pipeline (if they have that attribute already).

    Parameters
    ----------
    config
        Configurations of the experiment
    pipeline
        The entire pipeline

    Example
    -------
    if ``save_remote`` is set in the experiment configuration file,
    it will overwrite ``save_remote`` in all models that have that attribute.

    Returns
    -------
    None

    """

    for key, value in vars(config).items():
        if value is not None:
            for model in flatten(pipeline.children()):
                if hasattr(model, "config"):
                    if hasattr(model.config, key):
                        vars(model.config)[key] = value


def append_parent_path_and_id_(pipeline: Pipeline) -> None:
    entire_pipeline = pipeline.dict_children()

    def append(block, parent_path: str, id_with_prefix: str):
        block["obj"].parent_path = parent_path
        if not isinstance(block["obj"], DataSource):
            block["obj"].id += id_with_prefix

        if "children" in block:
            for i, child in enumerate(block["children"]):
                append(
                    child,
                    parent_path=f"{parent_path}/{block['obj'].id}",
                    id_with_prefix=f"{id_with_prefix}-{i}",
                )

    append(entire_pipeline, Const.output_pipelines_path, "")


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

        overwrite_model_configs_(experiment, self.pipeline)
        append_parent_path_and_id_(self.pipeline)

        self.run_path = f"{Const.output_runs_path}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"
        self.store = Store(data, labels, self.run_path)
        self.plugins = obligatory_plugins + plugins

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

        print("🤔 Evaluating entire pipeline")
        stats = evaluate(
            preds_probs, self.store, self.experiment.metrics, self.run_path
        )
        self.store.set_stats(Const.final_eval_name, stats)

        for plugin in self.plugins:
            plugin.print_me("on_run_end")
            _, _ = plugin.on_run_end(self.pipeline, self.store)
