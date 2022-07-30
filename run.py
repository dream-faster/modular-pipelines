from typing import List, Optional

from blocks.pipeline import Pipeline
from configs.constants import Const
from library.evaluation import calibration_metrics, classification_metrics
from library.examples.hate_speech import (cross_dataset_run_configs,
                                          ensemble_pipeline_hf,
                                          huggingface_baseline,
                                          preprocess_config,
                                          tweeteval_hate_speech_run_configs,
                                          vader)
from plugins import WandbConfig, WandbPlugin
from runner.runner import Runner
from type import Evaluators, Experiment, PreprocessConfig


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
                        preprocess_config=preprocess_config.get_configs(),
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
        
        vader,
        preprocess_config,
        project_id="hate-speech-detection",
        run_configs=cross_dataset_run_configs,
        metrics=metrics,
    )
