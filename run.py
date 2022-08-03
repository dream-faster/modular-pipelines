from typing import List

from configs.constants import Const

# from library.examples.hate_speech import all_cross_dataset_experiments
from library.examples.hate_speech_multi_hf import multi_hf_run_experiments
from plugins import WandbConfig, WandbPlugin
from runner.runner import Runner
from type import Experiment, StagingConfig, StagingNames
from utils.run_helpers import overwrite_preprocessing_configs_


def run(
    experiments: List[Experiment],
    staging_config: StagingConfig,
) -> None:

    for experiment in experiments:
        overwrite_preprocessing_configs_(experiment.dataloader, staging_config)

        experiment.dataloader.transform_()
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
                        preprocess_config=experiment.preprocessing_config.get_configs(),
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
            plugins=logger_plugins,
        )
        runner.run()


if __name__ == "__main__":
    prod_config = StagingConfig(
        name=StagingNames.prod, save_remote=True, log_remote=True, limit_dataset_to=None
    )

    dev_config = StagingConfig(
        name=StagingNames.prod,
        save_remote=False,
        log_remote=False,
        limit_dataset_to=100,
    )

    run(
        multi_hf_run_experiments,
        staging_config=dev_config,
    )
