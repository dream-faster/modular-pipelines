from collections.abc import Iterable
from typing import List, Union
import numpy as np
import pandas as pd


def flatten(input: List) -> List:
    if isinstance(input, Iterable):
        return [a for i in input for a in flatten(i)]
    else:
        return [input]


def remove_none(input: List) -> List:
    return [i for i in input if i is not None]


def convert_to_list(input: Union[np.ndarray, pd.Series, List]) -> List:
    if isinstance(input, np.ndarray):
        return input.tolist()
    if isinstance(input, pd.Series):
        return input.to_list()
    else:
        return input
