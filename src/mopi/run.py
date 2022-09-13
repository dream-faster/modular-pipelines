from re import A
from typing import List, Tuple

from mopi.constants import Const

from mopi.plugins import WandbConfig, WandbPlugin, OutputAnalyserPlugin
from mopi.runner.runner import Runner
from mopi.type import Experiment, StagingConfig, StagingNames, SourceTypes
from mopi.runner.utils import overwrite_preprocessing_configs_
from mopi.utils.json import dump_str

from mopi.blocks.io import export_pipeline, load_pipeline


def run(
    experiments: List[Experiment],
    staging_config: StagingConfig,
    pure_inference:bool = False
) -> List[Tuple[Experiment, "Store"]]:

    successes = []

    for experiment in experiments:

        overwrite_preprocessing_configs_(experiment.pipeline, staging_config)
        if (
            experiment.global_dataloader is not None
            and staging_config.limit_dataset_to is not None
        ):
            for (
                preprocessing_config
            ) in experiment.global_dataloader.preprocessing_configs:
                preprocessing_config.test_size = staging_config.limit_dataset_to
                preprocessing_config.train_size = staging_config.limit_dataset_to
                preprocessing_config.val_size = staging_config.limit_dataset_to

        experiment.save_remote = staging_config.save_remote
        experiment.log_remote = staging_config.log_remote

        logger_plugins = (
            [
                WandbPlugin(
                    WandbConfig(
                        project_id=experiment.project_name,
                        run_name=experiment.run_name + "---" + experiment.pipeline.id,
                        train=experiment.train,
                        delete_run=staging_config.delete_remote_log,
                        output_stats=True,
                    ),
                    run_config=experiment.get_configs(type_exclude=["Pipeline"]),
                )
            ]
            if staging_config.log_remote
            else []
        )
        runner = Runner(
            experiment,
            plugins=[OutputAnalyserPlugin()] + logger_plugins,
        )

        store = runner.run(pure_inference)
        successes.append((experiment, store))

        export_pipeline(
            experiment.pipeline.id,
            experiment.pipeline,
        )

    return successes


if __name__ == "__main__":
    from mopi.library.experiments.hate_speech import (
        all_merged_cross_experiments,
        all_tweeteval_crossexperiments,
    )

    prod_config = StagingConfig(
        name=StagingNames.prod,
        save_remote=False,
        log_remote=True,
        limit_dataset_to=None,
    )

    run(
        all_tweeteval_crossexperiments + all_merged_cross_experiments,
        staging_config=prod_config,
    )
