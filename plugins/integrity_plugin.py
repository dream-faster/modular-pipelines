from typing import List

from blocks.base import Block
from blocks.pipeline import Pipeline
from type import DataType, Hierarchy
from utils.flatten import flatten

from .base import Plugin


class IntegrityChecker(Plugin):
    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        print("    ┃  ├── 🆔 Verifying pipeline integrity")
        if not check_integrity(pipeline):
            raise Exception("Pipeline integrity check failed")
        else:
            print("    ┃  └── ✅ Integrity check passed")

        return pipeline


def check_integrity(pipeline: Pipeline) -> bool:
    hierarchy = pipeline.get_hierarchy()
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
            print(
                f"{previous_block.id}'s output type is {previous_block.outputType} and not {next_block.inputTypes} which {next_block.id} requires"
            )
            return False
        else:
            return True
    else:
        if (
            previous_block.outputType != next_block.inputTypes
            and next_block.inputTypes is not DataType.Any
        ):
            print(
                f"{previous_block.id}'s output type is {previous_block.outputType} and not {next_block.inputTypes} which {next_block.id} requires"
            )
            return False
        else:
            return True
