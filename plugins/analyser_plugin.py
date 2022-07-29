from pprint import pprint
from typing import List

from blocks.base import Block, DataSource, Element
from blocks.pipeline import Pipeline

from .base import Plugin


class PipelineAnalyser(Plugin):
    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        print("    ┃  └── 🗼 Hierarchy of Models:")
        full_pipeline = pipeline.children()

        def print_all(blocks: List[Element], indent="    ┃       "):

            for block in blocks:
                if isinstance(block, List):
                    indent += "    "
                    print_all(block, indent)
                    indent = indent[: len(indent) - len("    ")]
                elif isinstance(block, DataSource):
                    print(indent + " - " + block.id)
                    indent += "    "
                else:
                    print(indent + " - " + block.id)

        print_all(full_pipeline)

        return pipeline
