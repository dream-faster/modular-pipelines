from typing import List


from blocks.pipeline import Pipeline
from configs.constants import Const
from library.evaluation import calibration_metrics, classification_metrics
from library.examples.hate_speech import (
    cross_dataset_run_configs,
    ensemble_pipeline_hf,
    huggingface_baseline,
    preprocess_config,
    tweeteval_hate_speech_run_configs,
    vader,
)
from plugins import WandbConfig, WandbPlugin
from runner.runner import Runner
from type import Evaluators, PreprocessConfig, RunConfig



def run(
    pipeline: Pipeline,
    preprocess_config: PreprocessConfig,
    project_id: str,
    run_configs: List[RunConfig],
    metrics: Evaluators,
) -> None:

    for config in run_configs:
        logger_plugins = (
            [
                WandbPlugin(
                    WandbConfig(
                        project_id=project_id,
                        run_name=config.run_name + "-" + pipeline.id,
                        train=True,
                    ),
                    dict(
                        run_config=config.get_configs(),
                        preprocess_config=preprocess_config.get_configs(),
                        pipeline_configs=pipeline.get_configs(),
                    ),
                )
            ]
            if config.remote_logging
            else []
        )
        runner = Runner(
            config,
            pipeline,
            data={Const.input_col: config.dataset[Const.input_col]},
            labels=config.dataset[Const.label_col]
            if hasattr(config.dataset, Const.label_col)
            else None,

            evaluators=metrics,
            plugins=logger_plugins,

        )
        runner.run()


if __name__ == "__main__":

    metrics = classification_metrics + calibration_metrics

    run(
        vader,

        preprocess_config,
        project_id="hate-speech-detection",
        run_configs=cross_dataset_run_configs,
        metrics=metrics,
    )
