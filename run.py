from typing import List, Tuple

from constants import Const

from plugins import WandbConfig, WandbPlugin, OutputAnalyserPlugin
from runner.runner import Runner
from type import Experiment, StagingConfig, StagingNames, SourceTypes
from runner.utils import overwrite_preprocessing_configs_
from utils.json import dump_str
import traceback


def run(
    experiments: List[Experiment],
    staging_config: StagingConfig,
) -> List[Tuple[Experiment, "Store"]]:

    successes = []

    for experiment in experiments:

        overwrite_preprocessing_configs_(experiment.pipeline, staging_config)

        experiment.save_remote = staging_config.save_remote
        experiment.log_remote = staging_config.log_remote

        logger_plugins = (
            [
                WandbPlugin(
                    WandbConfig(
                        project_id=experiment.project_name,
                        run_name=experiment.run_name + "---" + experiment.pipeline.id,
                        train=experiment.train,
                    ),
                    dict(
                        run_config=experiment.get_configs(),
                        pipeline_config=experiment.pipeline.get_configs(),
                    ),
                )
            ]
            if staging_config.log_remote
            else []
        )
        runner = Runner(
            experiment,
            plugins=logger_plugins + [OutputAnalyserPlugin()],
        )

        store = runner.run()
        successes.append((experiment, store))

    return successes


if __name__ == "__main__":
    from library.experiments.hate_speech import (
        all_merged_cross_experiments,
        all_tweeteval_crossexperiments,
    )

    prod_config = StagingConfig(
        name=StagingNames.prod,
        save_remote=False,
        log_remote=False,
        limit_dataset_to=100,
    )

    run(
        all_tweeteval_crossexperiments + all_merged_cross_experiments,
        staging_config=prod_config,
    )
