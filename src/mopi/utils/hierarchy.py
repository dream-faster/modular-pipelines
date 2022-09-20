from typing import List
from mopi.blocks.base import Element, DataSource


def hierarchy_to_str(
    blocks: List[Element], indent="    ┃       ", string_to_return=""
) -> str:
    for block in blocks:
        if isinstance(block, List):
            indent += "    "
            string_to_return += hierarchy_to_str(block, indent)
            indent = indent[: len(indent) - len("    ")]
        elif isinstance(block, DataSource):
            string_to_return += indent + "-" + block.original_id + "\n"
            indent += "    "
        else:
            string_to_return += indent + "-" + block.id + "\n"

    return string_to_return
