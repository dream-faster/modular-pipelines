import os
import joblib
from typing import Union
from configs.constants import Const


def safe_loading(pipeline_id: str, id: str) -> "Model":
    path = f"{Const.output_path}/{pipeline_id}/{id}.pkl"

    if os.path.exists(path):
        print(f"    |- Loading model {pipeline_id}/{id}")
        with open(path, "rb") as f:
            return joblib.load(f)
    else:
        return None


def safe_saving(
    object: Union["Model", "Pipeline", None], pipeline_id: str, id: str
) -> None:
    path = f"{Const.output_path}/{pipeline_id}"
    if os.path.exists(path) is False:
        os.makedirs(path)

    print(f"| Saving model {pipeline_id}/{id}")
    with open(path + f"/{id}.pkl", "wb") as f:
        joblib.dump(object, f, compress=9)
