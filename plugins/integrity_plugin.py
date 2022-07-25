import logging
from blocks.pipeline import Pipeline

from .base import Plugin
from utils.flatten import flatten
from type import DataType


class IntegrityChecker(Plugin):
    def on_run_begin(self, pipeline: Pipeline) -> None:
        self.logger.info("🆔 Verifying pipeline integrity", extra=self.d)
        if not check_integrity(pipeline, self.logger, self.d):
            raise Exception("Pipeline integrity check failed")
        else:
            self.logger.info("✅ Integrity check passed", extra=self.d)

        return pipeline


def check_integrity(pipeline: Pipeline, logger: logging.Logger, d=dict[str]) -> bool:
    hierarchy = list(reversed(flatten(pipeline.children())))
    previous_block = hierarchy[0]
    for item in hierarchy[1:]:
        if isinstance(item, Pipeline):
            continue
        if isinstance(previous_block.inputTypes, list):
            if (
                item.outputType not in previous_block.inputTypes
                and DataType.Any not in previous_block.inputTypes
            ):
                logger.info(
                    f"{item.id}'s output type is {item.outputType} and not {previous_block.inputTypes} which {previous_block.id} requires",
                    extra=d,
                )
                return False
        else:
            if (
                item.outputType != previous_block.inputTypes
                and previous_block.inputTypes is not DataType.Any
            ):
                logger.info(
                    f"{item.id}'s output type is {item.outputType} and not {previous_block.inputTypes} which {previous_block.id} requires",
                    extra=d,
                )
                return False
        previous_block = item
    return True
