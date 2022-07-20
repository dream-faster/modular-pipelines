from pprint import pprint
from blocks.pipeline import Pipeline

from .base import Plugin


class PipelineAnalyser(Plugin):
    def on_run_begin(self, pipeline: Pipeline) -> None:
        super().on_run_begin(pipeline)

        print("🗼 Hierarchy of Models:")
        pprint(pipeline.children())
