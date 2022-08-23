from typing import List

from blocks.base import Block
from blocks.pipeline import Pipeline
from type import DataType, Hierarchy

from .base import Plugin
from utils.printing import logger


class IntegrityChecker(Plugin):
    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        logger.log("🆔 Verifying pipeline integrity", level=logger.levels.TWO)
        if not check_integrity(pipeline):
            raise Exception("Pipeline integrity check failed")
        else:
            logger.log("✅ Integrity check passed", level=logger.levels.TWO)

        return pipeline


def check_integrity(pipeline: Pipeline) -> bool:
    hierarchies = pipeline.get_hierarchy()
    for hierarchy in hierarchies:
        if hierarchy.children is not None:
            return __check_linear_block_integrity(hierarchy.children)
        else:
            return True


def __check_linear_block_integrity(hierarchies: List[Hierarchy]) -> bool:
    previous_block = hierarchies[0]
    if previous_block.children is not None:
        if __check_linear_block_integrity(previous_block.children) == False:
            return False
    for item in hierarchies[1:]:
        if item.children is not None:
            if __check_linear_block_integrity(item.children) == False:
                return False
        if check_if_types_correct(previous_block.obj, item.obj) == False:
            return False
        previous_block = item
    return True


def check_if_types_correct(previous_block: Block, next_block: Block) -> bool:
    if isinstance(next_block.inputTypes, List):
        if (
            previous_block.outputType not in next_block.inputTypes
            and DataType.Any not in next_block.inputTypes
        ):
            logger.log(
                f"{previous_block.id}'s output type is {previous_block.outputType} and not {next_block.inputTypes} which {next_block.id} requires",
                level=logger.levels.THREE,
            )
            return False
        else:
            return True
    else:
        if (
            previous_block.outputType != next_block.inputTypes
            and next_block.inputTypes is not DataType.Any
        ):
            logger.log(
                f"{previous_block.id}'s output type is {previous_block.outputType} and not {next_block.inputTypes} which {next_block.id} requires",
                level=logger.levels.THREE,
            )
            return False
        else:
            return True
