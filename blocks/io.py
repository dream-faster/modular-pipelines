import os
from typing import Union

import joblib

from configs.constants import Const
from utils.printing import PrintFormats

import textwrap


class PickleIO:
    def save(self) -> None:
        pickle_saving(
            self.model, f"{Const.output_pipelines_path}/{self.parent_path}", self.id
        )

    def load(self) -> None:
        model = pickle_loading(
            f"{Const.output_pipelines_path}/{self.parent_path}", self.id
        )
        if model is not None:
            self.model = model


def pickle_loading(parent_path: str, id: str) -> "Model":
    path = f"{parent_path}/{id}.pkl"

    if not os.path.exists(path):
        return None

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


def pickle_saving(
    object: Union["Model", "Pipeline", None], parent_path: str, id: str
) -> None:
    path = f"{parent_path}"
    if os.path.exists(path) is False:
        os.makedirs(path)

    print(
        textwrap.fill(
            f"Saving model {parent_path}/{PrintFormats.BOLD}{id}{PrintFormats.END}",
            initial_indent="    ┣━━━ ",
            subsequent_indent="    ┃        ",
            width=100,
        )
    )
    with open(path + f"/{id}.pkl", "wb") as f:
        joblib.dump(object, f, compress=9)