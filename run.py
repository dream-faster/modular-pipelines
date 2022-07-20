from configs.constants import Const
from data.dataloader import load_data
from runner.runner import Runner
from library.examples.hate_speech import (
    hate_speech_detection_pipeline,
    preprocess_config,
)
from library.evaluation import classification_metrics
from library.examples.all_transformations import all_transformations

from plugins import Plugin, WandbPlugin


train_dataset, test_dataset = load_data("data/original", preprocess_config)

pipeline = hate_speech_detection_pipeline()


runner = Runner(
    pipeline,
    data={"input": train_dataset[Const.input_col]},
    labels=train_dataset[Const.label_col],
    evaluators=classification_metrics,
    train=True,
    plugins=[Plugin(), WandbPlugin()],
)

runner.run()
