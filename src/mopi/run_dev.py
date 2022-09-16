from mopi.type import Experiment, StagingConfig, StagingNames

from mopi.run import run
from typing import List, Tuple


def run_dev(
    experiments: List[Experiment], pure_inference: bool = False
) -> List[Tuple[Experiment, "Pipeline", "Store"]]:

    dev_config = StagingConfig(
        name=StagingNames.dev,
        save_remote=False,
        log_remote=False,
        limit_dataset_to=60,
    )

    for experiment in experiments:
        experiment.project_name = "hate-speech-DEV"

    successes = run(
        experiments, staging_config=dev_config, pure_inference=pure_inference
    )

    return successes


if __name__ == "__main__":
    from mopi.library.experiments.hate_speech_baselines_trained import all_experiments

    run_dev(all_experiments)
