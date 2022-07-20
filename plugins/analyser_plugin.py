from pprint import pprint
from blocks.pipeline import Pipeline

from .base import Plugin
from utils.flatten import flatten
from type import DataType


class PipelineAnalyser(Plugin):
    def on_run_begin(self, pipeline: Pipeline) -> None:
        super().on_run_begin(pipeline)

        print("🗼 Hierarchy of Models:")
        pprint(pipeline.children())
