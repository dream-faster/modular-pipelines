from typing import Any, List, Optional
from enum import Enum
from copy import copy
import pandas as pd
import numpy as np


def is_custom_obj(obj: Any):
    if (
        type(obj).__module__ is not object.__module__
        and isinstance(obj, Enum) is False
        and hasattr(obj, "__dict__")
        and isinstance(obj, np.ndarray) is False
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

        elif isinstance(value, np.ndarray):
            obj_dict[key] = copy(pd.DataFrame(value).to_dict())

        elif isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            obj_dict[key] = copy(value.to_dict())

        elif is_custom_obj(value):
            obj_dict[key] = obj_to_dict(value)

    return obj_dict


def flatten(input: dict) -> dict:

    return pd.json_normalize(input, sep="_").to_dict(orient="records")[0]
