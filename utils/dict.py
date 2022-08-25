from typing import Any, List
from enum import Enum
from copy import copy


def is_custom_obj(obj: Any):
    if (
        type(obj).__module__ is not object.__module__
        and isinstance(obj, Enum) is False
        and hasattr(obj, "__dict__")
    ):
        return True
    else:
        return False


def list_to_dict(obj: List):
    return {
        el.id
        if hasattr(el, "id")
        else type(el).__name__: obj_to_dict(el)
        if is_custom_obj(el)
        else el
        for el in obj
    }


def obj_to_dict(obj: Any) -> dict:
    obj_dict = vars(copy(obj))

    for key, value in obj_dict.items():
        if isinstance(value, List):
            obj_dict[key] = list_to_dict(value)

        elif is_custom_obj(value):
            obj_dict[key] = obj_to_dict(value)

    return obj_dict
