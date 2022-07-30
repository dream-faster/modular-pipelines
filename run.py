from typing import List, Optional

from configs.constants import Const
from library.examples.hate_speech import all_cross_dataset_experiments
from plugins import WandbConfig, WandbPlugin
from runner.runner import Runner
from type import Experiment


def run(
    experiments: List[Experiment],
    save_remote: Optional[
        bool
    ] = None,  # If set True all models will try uploading (if configured), if set False it overwrites uploading of any models (even if configured)
    remote_logging: Optional[
        bool
    ] = None,  # Switches on and off all remote logging (eg.: wandb)
) -> None:

    for experiment in experiments:
        logger_plugins = (
            [
                WandbPlugin(
                    WandbConfig(
                        project_id=experiment.project_name,
                        run_name=experiment.run_name + "-" + experiment.pipeline.id,
                        train=True,
                    ),
                    dict(
                        run_config=experiment.get_configs(),
                        preprocess_config=experiment.preprocessing_config.get_configs(),
                        pipeline_configs=experiment.pipeline.get_configs(),
                    ),
                )
            ]
            if remote_logging
            else []
        )
        runner = Runner(
            experiment,
            data={Const.input_col: experiment.dataset[Const.input_col]},
            labels=experiment.dataset[Const.label_col],
            plugins=logger_plugins,
        )
        runner.run()


if __name__ == "__main__":

    run(
        all_cross_dataset_experiments,
        save_remote=False,
        remote_logging=True,
    )
