import os
from typing import Union

import joblib

from configs.constants import Const


def safe_loading(parent_path: str, id: str) -> "Model":
    path = f"{Const.output_pipelines_path}/{parent_path}/{id}.pkl"

    if os.path.exists(path):
        print(f"    ├ Loading model {parent_path}/{id}")
        with open(path, "rb") as f:
            return joblib.load(f)
    else:
        return None


def safe_saving(
    object: Union["Model", "Pipeline", None], parent_path: str, id: str
) -> None:
    path = f"{Const.output_pipelines_path}/{parent_path}"
    if os.path.exists(path) is False:
        os.makedirs(path)

    print(f"| Saving model {parent_path}/{id}")
    with open(path + f"/{id}.pkl", "wb") as f:
        joblib.dump(object, f, compress=9)
