from typing import Any, TypeVar
from copy import deepcopy

T = TypeVar("T")

def clone_and_set(obj: T, attribute_name: str, value: Any) -> T:
    new_obj = deepcopy(obj)
    setattr(new_obj, attribute_name, value)
    return new_obj
