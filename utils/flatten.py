from collections import Iterable
from typing import List


def flatten(input: List) -> List:
    if isinstance(input, Iterable):
        return [a for i in input for a in flatten(i)]
    else:
        return [input]

def remove_none(input: List) -> List:
    return [i for i in input if i is not None]