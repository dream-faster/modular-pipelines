from dataclasses import dataclass
from mopi.type import Experiment, StagingConfig, RunContext
from mopi.blocks.pipeline import Pipeline
from mopi.constants import Const
from mopi.utils.list import flatten
from mopi.blocks.base import DataSource
from mopi.data.dataloader import DataLoader
from mopi.type import Hierarchy
from copy import copy, deepcopy
from mopi.runner.store import Store


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
        for source_type in pipeline.get_datasource_types()
        for block in flatten(pipeline.children(source_type))
        if type(block) == DataSource
    ]
    if (
        hasattr(staging_config, "limit_dataset_to")
        and staging_config.limit_dataset_to is not None
    ):

        for datasource in datasources:
            datasource.dataloader.preprocessing_config.test_size = (
                staging_config.limit_dataset_to
            )
            datasource.dataloader.preprocessing_config.train_size = (
                staging_config.limit_dataset_to
            )
            datasource.dataloader.preprocessing_config.val_size = (
                staging_config.limit_dataset_to
            )

    # This is for overwriting exisiting keys in the preprocessing_config
    for key_sta, value_sta in vars(staging_config).items():
        if value_sta is not None:
            for datasource in datasources:
                preprocessing_config = datasource.dataloader.preprocessing_config
                for key_pre in vars(preprocessing_config).keys():
                    if key_pre == key_sta:
                        vars(preprocessing_config)[key_sta] = value_sta


def overwrite_dataloaders_(pipeline: Pipeline, dataloader: DataLoader) -> None:
    datasources = [
        block
        for source_type in pipeline.get_datasource_types()
        for block in flatten(pipeline.children(source_type))
        if type(block) == DataSource
    ]

    for datasource in datasources:
        datasource.dataloader = deepcopy(dataloader)
        datasource.id = copy(dataloader.path)


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
            for source_type in pipeline.get_datasource_types():
                for model in flatten(pipeline.children(source_type)):
                    if hasattr(model, "config"):
                        if hasattr(model.config, key):
                            vars(model.config)[key] = value


def handle_datasource_id_(block, seen_datasources: dict) -> None:
    if block.obj.id not in seen_datasources.keys():
        original_block_obj_id = block.obj.id
        block.obj.id = f"datasource_{len(seen_datasources)}"
        seen_datasources[original_block_obj_id] = block.obj.id
    else:
        block.obj.id = seen_datasources[block.obj.id]


def append_parent_path_and_id_(pipeline: Pipeline, mask: bool = False) -> None:
    """
    Appends TWO values to each object in the pipeline:
        - ``id``: adds a unique integer as a suffix at the end of each id
        - ``parent_path``: adds the path to the object (used for saving)

    Parameters
    ----------
    pipeline
        The entire pipeline
    mask
        Wether it should mask out datasources with generic names

    Returns
    -------
    None

    """

    hierarchies = [
        pipeline.get_hierarchy(source_types)
        for source_types in pipeline.get_datasource_types()
    ]

    for hierarchy in hierarchies:
        seen_datasources = {}

        def append(block, parent_path: str, id_with_prefix: str):
            block.obj.parent_path = parent_path
            if not isinstance(block.obj, DataSource):
                block.obj.id += id_with_prefix
            elif mask:
                handle_datasource_id_(block, seen_datasources)

            if hasattr(block, "children") and block.children is not None:
                for i, child in enumerate(block.children):
                    append(
                        child,
                        parent_path=f"{parent_path}/{block.obj.id}",
                        id_with_prefix=f"{id_with_prefix}-{i}",
                    )

        append(hierarchy, pipeline.run_context.project_name, "")


def add_experiment_config_to_blocks_(
    pipeline: Pipeline, experiment: Experiment
) -> None:
    for source_type in pipeline.get_datasource_types():
        for model in flatten(pipeline.children(source_type)):
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
        for source_type in pipeline.get_datasource_types()
        for block in flatten(pipeline.children(source_type))
        if type(block) == DataSource
    ]

    for datasource in datasources:
        datasource.category = experiment.dataset_category
