import os
from typing import Union

import joblib

from mopi.constants import Const

from mopi.utils.printing import logger
from mopi.utils.list import flatten
import dill as pickle


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
            logger.log(
                f"✅ Loaded model",
                level=logger.levels.TWO,
            )


def pickle_loading(parent_path: str, id: str) -> "Model":
    path = f"{parent_path}/{id}.pkl"

    if not os.path.exists(path):
        return None

    logger.log(
        f"Loading model {parent_path}/{logger.formats.BOLD}{id}{logger.formats.END}",
        mode=logger.modes.MULTILINE,
        level=logger.levels.ONE,
    )

    with open(path, "rb") as f:
        return joblib.load(f)


def pickle_saving(
    object: Union["Model", "Pipeline", None], parent_path: str, id: str
) -> None:
    path = f"{parent_path}"
    if os.path.exists(path) is False:
        os.makedirs(path)

    logger.log(
        f"Saving model {parent_path}/{logger.formats.BOLD}{id}{logger.formats.END}",
        mode=logger.modes.MULTILINE,
    )
    with open(path + f"/{id}.pkl", "wb") as f:
        joblib.dump(object, f, compress=9)


def export_pipeline(name: str, pipeline: "Pipeline") -> None:

    blocks = [
        block
        for source_type in pipeline.get_datasource_types()
        for block in flatten(pipeline.children(source_type))
    ]

    for block in blocks:
        if hasattr(block, "model") and block.model is not None:
            block.model = None
        if (
            hasattr(block, "dataloader")
            and hasattr(block.dataloader, "data")
            and block.dataloader.data is not None
        ):
            block.dataloader.data = None

    if os.path.exists(Const.output_pipelines_path) is False:
        os.makedirs(Const.output_pipelines_path)

    with open(f"{Const.output_pipelines_path}/{name}", "wb") as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def load_pipeline(name: str) -> "Pipeline":
    with open(f"{Const.output_pipelines_path}/{name}", "rb") as handle:
        pipeline = pickle.load(handle)

    return pipeline
