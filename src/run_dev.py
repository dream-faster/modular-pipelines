from library.experiments.hate_speech_multi_objective import multi_objective_experiments
from library.experiments.hate_speech_baselines import all_experiments
from src.type import Experiment

from type import StagingConfig, StagingNames


from run import run
from typing import List


def run_dev(experiments: List[Experiment]) -> None:

    dev_config = StagingConfig(
        name=StagingNames.dev,
        save_remote=False,
        log_remote=True,
        limit_dataset_to=60,
    )

    for experiment in experiments:
        experiment.project_name = "hate-speech-DEV"

    run(
        experiments,
        staging_config=dev_config,
    )


if __name__ == "__main__":
    run_dev(all_experiments)
