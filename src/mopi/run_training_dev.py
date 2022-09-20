from mopi.type import Experiment, StagingConfig, StagingNames

from mopi.run_training import run_training
from typing import List, Tuple


def run_training_dev(
    experiments: List[Experiment],
) -> List[Tuple[Experiment, "Pipeline", "Store"]]:

    dev_config = StagingConfig(
        name=StagingNames.dev,
        save_remote=False,
        log_remote=False,
        limit_dataset_to=60,
    )

    for experiment in experiments:
        experiment.project_name = "hate-speech-DEV"

    successes = run_training(experiments, staging_config=dev_config)

    return successes


if __name__ == "__main__":
    from mopi.library.experiments.hate_speech_baselines import all_experiments

    run_training_dev(all_experiments)
