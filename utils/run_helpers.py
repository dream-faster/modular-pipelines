from type import Experiment
from blocks.pipeline import Pipeline
from configs.constants import Const
from .flatten import flatten
from blocks.base import DataSource


def overwrite_model_configs_(config: Experiment, pipeline: Pipeline) -> None:
    """
    Takes global config values and overwrites the config of
    each model in the pipeline (if they have that attribute already).

    Parameters
    ----------
    config
        Configurations of the experiment (global settings of the experiment/run)
    pipeline
        The entire pipeline object

    Example
    -------
    if ``save_remote`` is set in the experiment configuration file,
    it will overwrite ``save_remote`` in all models that have that attribute.

    Returns
    -------
    None

    """

    for key, value in vars(config).items():
        if value is not None:
            for model in flatten(pipeline.children()):
                if hasattr(model, "config"):
                    if hasattr(model.config, key):
                        vars(model.config)[key] = value


def append_parent_path_and_id_(pipeline: Pipeline) -> None:
    """
    Appends two values to each object in the pipeline:
        - ``id``: adds a unique integer as a suffix at the end of each id
        - ``parent_path``: adds the path to the object (used for saving)

    Parameters
    ----------
    pipeline
        The entire pipeline

    Returns
    -------
    None

    """

    entire_pipeline = pipeline.dict_children()

    def append(block, parent_path: str, id_with_prefix: str):
        block["obj"].parent_path = parent_path
        if not isinstance(block["obj"], DataSource):
            block["obj"].id += id_with_prefix

        if "children" in block:
            for i, child in enumerate(block["children"]):
                append(
                    child,
                    parent_path=f"{parent_path}/{block['obj'].id}",
                    id_with_prefix=f"{id_with_prefix}-{i}",
                )

    append(entire_pipeline, Const.output_pipelines_path, "")
