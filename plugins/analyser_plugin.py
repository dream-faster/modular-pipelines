from pprint import pprint
from typing import List
from blocks.base import Block, Element
from blocks.pipeline import Pipeline

from .base import Plugin


class PipelineAnalyser(Plugin):
    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        self.logger.info("🗼 Hierarchy of Models:", extra=self.d)
        full_pipeline = pipeline.children()

        def print_all(blocks: List[Element], indent="    ┃       "):

            for block in blocks:
                if isinstance(block, List):
                    indent += "    "
                    print_all(block, indent)
                else:
                    self.logger.info(indent + " - " + block.id, extra=self.d)

        print_all(full_pipeline)

        return pipeline
