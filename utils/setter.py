from typing import Any, TypeVar
from copy import deepcopy
from typing_extensions import Self

T = TypeVar("T")

class Settable:
    def set_attr(self, key: str, value) -> Self:
        new_obj = deepcopy(self)
        setattr(new_obj, key, value)
        return new_obj


def clone_and_set(obj: T, dicticionary: dict) -> T:
    new_obj = deepcopy(obj)
    for key, value in dicticionary:
        setattr(new_obj, key, value)
    return new_obj
