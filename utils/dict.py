from typing import Any, List, Optional, Tuple, Union
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


def list_to_dict(obj: Union[List, Tuple], type_exclude: Optional[str] = None):
    return {
        el.id
        if hasattr(el, "id")
        else type(el).__name__: obj_to_dict(el, type_exclude=type_exclude)
        if is_custom_obj(el)
        else el
        for el in obj
    }


def obj_to_dict(obj: Any, type_exclude: Optional[str] = None) -> dict:
    obj_dict = vars(copy(obj))

    for key in list(obj_dict.keys()):
        value = obj_dict[key]

        if type_exclude is not None and type(value).__name__ == type_exclude:
            del obj_dict[key]
            continue

        if isinstance(value, (List, Tuple)):
            obj_dict[key] = list_to_dict(value, type_exclude=type_exclude)

        elif isinstance(value, np.ndarray):
            obj_dict[key] = copy(pd.DataFrame(value).to_dict())

        elif isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            obj_dict[key] = copy(value.to_dict())

        elif is_custom_obj(value):
            obj_dict[key] = obj_to_dict(value, type_exclude=type_exclude)

    return obj_dict


def flatten(input: dict) -> dict:

    return pd.json_normalize(input, sep="_").to_dict(orient="records")[0]
