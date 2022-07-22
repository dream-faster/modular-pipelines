from pprint import pprint
from typing import List
from blocks.base import Block, Element
from blocks.pipeline import Pipeline

from .base import Plugin


class PipelineAnalyser(Plugin):
    def on_run_begin(self, pipeline: Pipeline) -> None:
        super().on_run_begin(pipeline)

        print("    ┃  └── 🗼 Hierarchy of Models:")
        full_pipeline = pipeline.children()

        def print_all(blocks: List[Element], indent="    ┃       "):

            for block in blocks:
                if isinstance(block, List):
                    indent += "    "
                    print_all(block, indent)
                else:
                    print(indent + " - " + block.id)

        print_all(full_pipeline)

        # pprint(pipeline.children())
