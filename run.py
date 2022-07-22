from datasets import Dataset
from configs.constants import Const
from data.dataloader import load_data
from runner.runner import Runner
from library.examples.hate_speech import (
    hate_speech_detection_pipeline,
    preprocess_config,
)
from library.evaluation import classification_metrics

from blocks.pipeline import Pipeline
from typing import Tuple
from plugins import WandbPlugin, WandbConfig


hate_speech_data = load_data("data/original", preprocess_config)


def run(pipeline: Pipeline, data: Tuple[Dataset, Dataset]) -> None:
    train_dataset, test_dataset = data

    train_runner = Runner(
        pipeline,
        data={"input": train_dataset[Const.input_col]},
        labels=train_dataset[Const.label_col],
        evaluators=classification_metrics,
        train=True,
        plugins=[
            # WandbPlugin(
            #     WandbConfig(project_id="hate-speech-detection"), pipeline.get_configs()
            # )
        ],
    )
    train_runner.run()

    test_runner = Runner(
        pipeline,
        data={"input": test_dataset[Const.input_col]},
        labels=test_dataset[Const.label_col],
        evaluators=classification_metrics,
        train=False,
        plugins=[],
    )
    test_runner.run()


if __name__ == "__main__":
    run(hate_speech_detection_pipeline(), hate_speech_data)
