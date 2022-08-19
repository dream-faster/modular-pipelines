from typing import Any, TypeVar
from copy import deepcopy

T = TypeVar("T")

def clone_and_set(obj: T, dicticionary: dict) -> T:
    new_obj = deepcopy(obj)
    for key, value in dicticionary:
        setattr(new_obj, key, value)
    return new_obj
