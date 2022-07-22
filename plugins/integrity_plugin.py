from blocks.pipeline import Pipeline

from .base import Plugin
from utils.flatten import flatten
from type import DataType


class IntegrityChecker(Plugin):
    def on_run_begin(self, pipeline: Pipeline) -> Pipeline:
        super().on_run_begin(pipeline)

        print("    ┃  ├── 🆔 Verifying pipeline integrity")
        if not check_integrity(pipeline):
            raise Exception("Pipeline integrity check failed")
        else:
            print("    ┃  └── ✅ Integrity check passed")

        return pipeline


def check_integrity(pipeline: Pipeline) -> bool:
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
                print(
                    f"{item.id}'s output type is {item.outputType} and not {previous_block.inputTypes} which {previous_block.id} requires"
                )
                return False
        else:
            if (
                item.outputType != previous_block.inputTypes
                and previous_block.inputTypes is not DataType.Any
            ):
                print(
                    f"{item.id}'s output type is {item.outputType} and not {previous_block.inputTypes} which {previous_block.id} requires"
                )
                return False
        previous_block = item
    return True
