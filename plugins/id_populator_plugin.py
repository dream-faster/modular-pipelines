from typing import List, Union

from blocks.base import Block, DataSource, Element
from blocks.pipeline import Pipeline
from configs.constants import Const

from .base import Plugin


class IDPopulator(Plugin):
    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        print("    ┃  └── 🆔 Appending hierarchy locations to IDs.")
        pipeline = append_hierarchy_id(pipeline)

        print("    ┃  └── 🌳 Adding hierarchy paths to blocks.")
        pipeline = append_path_id(pipeline)

        return pipeline


def append_hierarchy_id(pipeline: Pipeline) -> Pipeline:
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


def append_path_id(pipeline: Pipeline) -> Pipeline:
    entire_pipeline = pipeline.dict_children()

    def append_id(block, pipeline_id: str):
        block["obj"].pipeline_id = f"{pipeline_id}"

        if "children" in block:
            for child in block["children"]:
                append_id(child, f"{pipeline_id}/{block['name']}")

    append_id(entire_pipeline, Const.output_pipelines_path)
    return pipeline
