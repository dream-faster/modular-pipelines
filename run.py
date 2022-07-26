from datasets.arrow_dataset import Dataset
from configs.constants import Const
from data.dataloader import load_data
from runner.runner import Runner
from library.examples.hate_speech import (
    ensemble_pipeline,
    preprocess_config,
)
from library.evaluation import classification_metrics

from blocks.pipeline import Pipeline
from typing import Tuple
from plugins import WandbPlugin, WandbConfig
from type import PreprocessConfig


hate_speech_data = load_data("data/original", preprocess_config)


def run(
    pipeline: Pipeline,
    data: Tuple[Dataset, Dataset],
    preprocess_config: PreprocessConfig,
    project_id: str,
) -> None:
    train_dataset, test_dataset = data

    train_runner = Runner(
        pipeline,
        data={"input": train_dataset[Const.input_col]},
        labels=train_dataset[Const.label_col],
        evaluators=classification_metrics,
        train=True,
        plugins=[
            WandbPlugin(
                WandbConfig(
                    project_id=project_id,
                    run_name=pipeline.id,
                    train=True,
                ),
                dict(pipeline.get_configs(), preprocess_config=vars(preprocess_config)),
            )
        ],
    )
    train_runner.run()

    test_runner = Runner(
        pipeline,
        data={"input": test_dataset[Const.input_col]},
        labels=test_dataset[Const.label_col],
        evaluators=classification_metrics,
        train=False,
        plugins=[
            WandbPlugin(
                WandbConfig(project_id=project_id, run_name=pipeline.id, train=False),
                dict(pipeline.get_configs(), preprocess_config=vars(preprocess_config)),
            )
        ],
    )
    test_runner.run()


if __name__ == "__main__":
    run(
        ensemble_pipeline,
        hate_speech_data,
        preprocess_config,
        project_id="hate-speech-detection",
    )
