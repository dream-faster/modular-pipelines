from library.experiments.hate_speech_multi_objective import multi_objective_experiments
from library.experiments.hate_speech_baselines import all_experiments

from type import StagingConfig, StagingNames


from run import run

experiments_list = all_experiments


dev_config = StagingConfig(
    name=StagingNames.dev,
    save_remote=False,
    log_remote=True,
    limit_dataset_to=60,
)

for experiment in experiments_list:
    experiment.project_name = "hate-speech-DEV"


run(
    experiments_list,
    staging_config=dev_config,
)
