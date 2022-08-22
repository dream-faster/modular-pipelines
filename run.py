from typing import List

from configs.constants import Const

from library.experiments.hate_speech import (
    all_tweeteval_experiments,
    all_tweeteval_cross_experiments,
    all_merged_cross_experiments,
)
from library.experiments.hate_speech_multi_hf import multi_hf_run_experiments
from library.experiments.hate_speech_perspective import perspective_experiments
from plugins import WandbConfig, WandbPlugin, OutputAnalyserPlugin
from runner.runner import Runner
from type import Experiment, StagingConfig, StagingNames
from utils.run_helpers import overwrite_preprocessing_configs_
from utils.json import dump_str


def run(
    experiments: List[Experiment],
    staging_config: StagingConfig,
) -> None:

    failed: List[Experiment] = []

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
        try:
            runner.run()
        except Exception as e:
            print(
                f"Run {experiment.project_name} - {experiment.run_name} - {experiment.pipeline.id} failed, due to"
                + f"\n{e}"
            )
            failed.append(experiment)

    if len(failed) > 0:
        failed_ids = "\n".join(
            [f"{exp.project_name}-{exp.run_name}-{exp.pipeline.id}" for exp in failed]
        )
        dump_str(failed_ids, "output/failed_runs.txt")


if __name__ == "__main__":
    prod_config = StagingConfig(
        name=StagingNames.prod,
        save_remote=False,
        log_remote=True,
        limit_dataset_to=None,
    )

    dev_config = StagingConfig(
        name=StagingNames.dev,
        save_remote=False,
        log_remote=False,
        limit_dataset_to=60,
    )

    run(
        all_tweeteval_experiments,
        # + all_tweeteval_cross_experiments
        # + all_merged_cross_experiments,
        staging_config=prod_config,
    )
