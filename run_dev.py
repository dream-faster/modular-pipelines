from library.experiments.hate_speech_multi_objective import multi_objective_experiments
from library.experiments.hate_speech_baselines import all_experiments

from type import StagingConfig, StagingNames


from run import run


dev_config = StagingConfig(
    name=StagingNames.dev,
    save_remote=False,
    log_remote=False,
    limit_dataset_to=60,
)

for experiment in all_experiments:
    experiment.project_name = "hate-speech-DEV"


run(
    all_experiments,
    staging_config=dev_config,
)
