from library.experiments.hate_speech_multi_type_hf import multi_type_hf_run_experiments

from type import StagingConfig, StagingNames


from run import run


dev_config = StagingConfig(
    name=StagingNames.dev,
    save_remote=False,
    log_remote=False,
    limit_dataset_to=60,
)

run(
    multi_type_hf_run_experiments,
    staging_config=dev_config,
)
