import datetime
from copy import deepcopy
from typing import Dict, List, Optional, Union

import pandas as pd
from blocks.base import Block, DataSource, Element
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


def overwrite_model_configs(config: RunConfig, pipeline: Pipeline) -> Pipeline:
    for key, value in vars(config).items():
        if value is not None:
            for model in flatten(pipeline.children()):
                if hasattr(model, "config"):
                    if hasattr(model.config, key):
                        vars(model.config)[key] = value

    return pipeline


def add_position_to_block_names(pipeline: Pipeline) -> Pipeline:
    entire_pipeline = pipeline.children()

    def add_position(block: Union[List[Element], Element], position: int, prefix: str):
        if isinstance(block, List):
            if position > 0:
                prefix += f"{position - 1}-"
            for i, child in enumerate(block):
                add_position(child, i, prefix)
        elif not isinstance(block, DataSource):
            block.id += f"{prefix}{position}"

    add_position(entire_pipeline, 1, "-")

    return pipeline


def append_pipeline_id(pipeline: Pipeline) -> Pipeline:
    entire_pipeline = pipeline.dict_children()

    def append_id(block, pipeline_id: str):
        block["obj"].pipeline_id = pipeline_id

        if "children" in block:
            for child in block["children"]:
                append_id(child, f"{pipeline_id}/{block['name']}/{child['name']}")

    append_id(entire_pipeline, Const.output_pipelines_path)
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
        self.pipeline = add_position_to_block_names(self.pipeline)
        self.pipeline = append_pipeline_id(self.pipeline)

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

        print("🤔 Evaluating entire pipeline")
        stats = evaluate(preds_probs, self.store, self.evaluators, self.run_path)
        self.store.set_stats(Const.final_eval_name, stats)

        for plugin in self.plugins:
            plugin.print_me("on_run_end")
            _, _ = plugin.on_run_end(self.pipeline, self.store)
