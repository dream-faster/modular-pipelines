from typing import List

from configs.constants import Const

from library.experiments.hate_speech import (
    all_cross_dataset_experiments,
    all_tweeteval_hate_speech_experiments,
)
from library.experiments.hate_speech_multi_hf import multi_hf_run_experiments
from plugins import WandbConfig, WandbPlugin, OutputAnalyserPlugin
from runner.runner import Runner
from type import Experiment, StagingConfig, StagingNames
from utils.run_helpers import overwrite_preprocessing_configs_


def run(
    experiments: List[Experiment],
    staging_config: StagingConfig,
) -> None:

    for experiment in experiments:

        overwrite_preprocessing_configs_(experiment.dataloader, staging_config)
        data = experiment.dataloader.load(experiment.dataset_category)

        experiment.save_remote = staging_config.save_remote
        experiment.log_remote = staging_config.log_remote

        logger_plugins = (
            [
                WandbPlugin(
                    WandbConfig(
                        project_id=experiment.project_name,
                        run_name=experiment.run_name + "-" + experiment.pipeline.id,
                        train=experiment.train,
                    ),
                    dict(
                        run_config=experiment.get_configs(),
                        preprocess_config=experiment.dataloader.preprocessing_configs[
                            0
                        ].get_configs(),
                        pipeline_configs=experiment.pipeline.get_configs(),
                    ),
                )
            ]
            if staging_config.log_remote
            else []
        )
        runner = Runner(
            experiment,
            data={Const.input_col: data[Const.input_col]},
            labels=data[Const.label_col],
            plugins=logger_plugins + [OutputAnalyserPlugin()],
        )
        runner.run()


if __name__ == "__main__":
    prod_config = StagingConfig(
        name=StagingNames.prod, save_remote=True, log_remote=True, limit_dataset_to=None
    )

    dev_config = StagingConfig(
        name=StagingNames.dev,
        save_remote=False,
        log_remote=False,
        limit_dataset_to=1000,
    )

    run(
        all_tweeteval_hate_speech_experiments,
        staging_config=dev_config,
    )
