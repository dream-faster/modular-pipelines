from typing import List, Tuple

from mopi.plugins import WandbConfig, WandbPlugin, OutputAnalyserPlugin
from mopi.runner.runner import Runner
from mopi.type import Experiment, StagingConfig, StagingNames
from mopi.runner.utils import overwrite_preprocessing_configs_

from mopi.blocks.io import export_pipeline


def run_training(
    experiments: List[Experiment],
    staging_config: StagingConfig,
    save_entire_pipeline: bool = False,
) -> List[Tuple[Experiment, "Pipeline", "Store"]]:

    successes = []

    for experiment in experiments:

        overwrite_preprocessing_configs_(experiment.pipeline, staging_config)
        if (
            experiment.global_dataloader is not None
            and staging_config.limit_dataset_to is not None
        ):
            experiment.global_dataloader.preprocessing_config.test_size = (
                staging_config.limit_dataset_to
            )
            experiment.global_dataloader.preprocessing_config.train_size = (
                staging_config.limit_dataset_to
            )
            experiment.global_dataloader.preprocessing_config.val_size = (
                staging_config.limit_dataset_to
            )

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
                    run_config=experiment.get_configs(
                        type_exclude=["Pipeline"], key_exclude=["global_dataloader"]
                    ),
                )
            ]
            if staging_config.log_remote
            else []
        )
        runner = Runner(
            experiment,
            plugins=[OutputAnalyserPlugin()] + logger_plugins,
        )

        store, pipeline, unloaded_pipeline = runner.train_test()

        successes.append((experiment, pipeline, store))

        if save_entire_pipeline is True and experiment.train is True:
            export_pipeline(
                unloaded_pipeline.id,
                unloaded_pipeline,
            )

    return successes


if __name__ == "__main__":
    from mopi.library.experiments.hate_speech import (
        all_merged_cross_experiments,
        all_tweeteval_crossexperiments,
        all_dynahate_cross_experiments,
        all_offensive_cross_experiments,
    )

    prod_config = StagingConfig(
        name=StagingNames.prod,
        save_remote=False,
        log_remote=True,
        limit_dataset_to=None,
    )

    run_training(
        all_tweeteval_crossexperiments
        + all_dynahate_cross_experiments
        + all_offensive_cross_experiments
        + all_merged_cross_experiments,
        staging_config=prod_config,
    )
