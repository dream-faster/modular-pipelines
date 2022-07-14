from configs.constants import Const

from data.dataloader import load_data
from configs.config import global_preprocess_config
from runner.run import train_pipeline
from library import all_transformations

train_dataset, test_dataset = load_data("data/original", global_preprocess_config)

train_pipeline(
    all_transformations(), {"input": train_dataset}, train_dataset[Const.label_col]
)
