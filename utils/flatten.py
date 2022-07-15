from collections import Iterable
from typing import List


def flatten(input: List) -> List:
    if isinstance(input, Iterable):
        return [a for i in input for a in flatten(i)]
    else:
        return [input]
