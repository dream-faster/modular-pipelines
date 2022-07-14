from configs.constants import Const

from data.dataloader import load_data
from runner.run import run_pipeline
from library.examples.hate_speech import (
    hate_speech_detection_pipeline,
    preprocess_config,
)
from library.examples.all_transformations import all_transformations


train_dataset, test_dataset = load_data("data/original", preprocess_config)

run_pipeline(
    hate_speech_detection_pipeline(),
    {"input": train_dataset[Const.input_col]},
    train_dataset[Const.label_col],
    train=False,
)
