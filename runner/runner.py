import datetime
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
    for key, value in vars(config).items():
        if value is not None:
            for model in flatten(pipeline.children()):
                if hasattr(model, "config"):
                    if hasattr(model.config, key):
                        vars(model.config)[key] = value


def append_parent_path_and_id(pipeline: Pipeline) -> None:
    entire_pipeline = pipeline.dict_children()

    blocks_encountered = []

    def append(block, parent_path: str, id_with_prefix: str):
        block["obj"].parent_path = f"{parent_path}"
        block["obj"].id += f"{id_with_prefix}"
        blocks_encountered.append(id(block["obj"]))

        if "children" in block:
            for i, child in enumerate(block["children"]):
                child_id = (
                    f"{id_with_prefix}-{i}"
                    if id(child["obj"]) not in blocks_encountered
                    else f"{id_with_prefix}+{i}"
                )
                append(
                    child,
                    parent_path=f"{parent_path}/{block['obj'].id}",
                    id_with_prefix=child_id,
                )

    append(entire_pipeline, Const.output_pipelines_path, "")


def rename_input_id_(
    pipeline: Pipeline, data: Dict[str, Union[pd.Series, List]]
) -> None:
    datasources = [
        block
        for block in flatten(pipeline.children())
        if type(block) is DataSource
        and block.id.split("-")[0].split("+")[0] == Const.input_col
    ]

    data[datasources[0].id] = data.pop(Const.input_col)


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
        append_parent_path_and_id(self.pipeline)
        rename_input_id_(self.pipeline, data)
        
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
