from pprint import pprint
from typing import List

from mopi.blocks.base import Block, DataSource, Element
from mopi.blocks.pipeline import Pipeline

from .base import Plugin
from mopi.utils.printing import logger
from mopi.constants import Const
from mopi.utils.hierarchy import hierarchy_to_str


class PipelineAnalyser(Plugin):
    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        logger.log("🗼 Hierarchy of Models:", level=logger.levels.TWO)
        source_types = pipeline.get_datasource_types()
        for source_type in source_types:
            if len(source_types) > 1:
                logger.log(
                    f"{logger.formats.BOLD}{source_type}{logger.formats.END}",
                    level=logger.levels.THREE,
                )
            full_pipeline = pipeline.children(source_type)

            print(hierarchy_to_str(full_pipeline))

        return pipeline
