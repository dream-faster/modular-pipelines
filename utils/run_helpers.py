from dataclasses import dataclass
from type import Experiment, StagingConfig, RunContext
from blocks.pipeline import Pipeline
from constants import Const
from .list import flatten
from blocks.base import DataSource
from data.dataloader import DataLoader
from type import Hierarchy
from copy import copy
from runner.store import Store


def overwrite_preprocessing_configs_(
    pipeline: Pipeline, staging_config: StagingConfig
) -> None:
    """
    Takes global config values and overwrites the preprocessing config according to the logic of the parameters

    Parameters
    ----------
    pipeline
        The entire pipeline the experiment is run on.
    stagingConfig
        The global stagingconfiguration

    Returns
    -------
    None

    """
    datasources = [
        block
        for pipeline_ in [pipeline.children("fit"), pipeline.children("predict")]
        for block in flatten(pipeline_)
        if type(block) == DataSource
    ]
    if (
        hasattr(staging_config, "limit_dataset_to")
        and staging_config.limit_dataset_to is not None
    ):

        for datasource in datasources:
            for preprocessing_config in datasource.dataloader.preprocessing_configs:
                preprocessing_config.test_size = staging_config.limit_dataset_to
                preprocessing_config.train_size = staging_config.limit_dataset_to
                preprocessing_config.val_size = staging_config.limit_dataset_to

    # This is for overwriting exisiting keys in the preprocessing_config
    for key_sta, value_sta in vars(staging_config).items():
        if value_sta is not None:
            for datasource in datasources:
                for preprocessing_config in datasource.dataloader.preprocessing_configs:
                    for key_pre in vars(preprocessing_config).keys():
                        if key_pre == key_sta:
                            vars(preprocessing_config)[key_sta] = value_sta


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
            for pipeline_ in [pipeline.children("fit"), pipeline.children("predict")]:
                for model in flatten(pipeline_):
                    if hasattr(model, "config"):
                        if hasattr(model.config, key):
                            vars(model.config)[key] = value


def append_parent_path_and_id_(pipeline: Pipeline) -> None:
    """
    Appends TWO values to each object in the pipeline:
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

    entire_pipelines = [
        pipeline.get_hierarchy("fit"),
        pipeline.get_hierarchy("predict"),
    ]

    for entire_pipeline in entire_pipelines:

        def append(block, parent_path: str, id_with_prefix: str):
            block.obj.parent_path = parent_path
            if not isinstance(block.obj, DataSource):
                block.obj.id += id_with_prefix

            if hasattr(block, "children"):
                if block.children is not None:
                    for i, child in enumerate(block.children):
                        append(
                            child,
                            parent_path=f"{parent_path}/{block.obj.id}",
                            id_with_prefix=f"{id_with_prefix}-{i}",
                        )

        append(entire_pipeline, pipeline.run_context.project_name, "")


def add_experiment_config_to_blocks_(
    pipeline: Pipeline, experiment: Experiment
) -> None:
    for pipeline_ in [pipeline.children("fit"), pipeline.children("predict")]:
        for model in flatten(pipeline_):
            model.run_context = RunContext(
                project_name=experiment.project_name,
                run_name=experiment.run_name,
                train=experiment.train,
            )


def add_split_category_to_datasource_(
    pipeline: Pipeline, experiment: Experiment
) -> None:

    datasources = [
        block
        for pipeline_ in [pipeline.children("fit"), pipeline.children("predict")]
        for block in flatten(pipeline_)
        if type(block) == DataSource
    ]

    for datasource in datasources:
        datasource.category = experiment.dataset_category
