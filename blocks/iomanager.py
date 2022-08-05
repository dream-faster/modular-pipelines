import os
from typing import Union

import joblib

from configs.constants import Const
from utils.printing import PrintFormats

import textwrap


def safe_loading(parent_path: str, id: str) -> "Model":
    path = f"{parent_path}/{id}.pkl"

    if os.path.exists(path):

        print(
            textwrap.fill(
                f"Loading model {parent_path}/{PrintFormats.BOLD}{id}{PrintFormats.END}",
                initial_indent="    ┣━━━ ",
                subsequent_indent="    ┃        ",
                width=100,
            )
        )
        with open(path, "rb") as f:
            return joblib.load(f)
    else:
        return None


def safe_saving(
    object: Union["Model", "Pipeline", None], parent_path: str, id: str
) -> None:
    path = f"{parent_path}"
    if os.path.exists(path) is False:
        os.makedirs(path)

    print(f"| Saving model {parent_path}/{id}")
    with open(path + f"/{id}.pkl", "wb") as f:
        joblib.dump(object, f, compress=9)
