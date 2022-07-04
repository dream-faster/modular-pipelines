from typing import List
from model.huggingface import HuggingfaceModel

from model.sklearn import SKLearnModel
from data.dataloader import load_data
from config import (
    global_preprocess_config,
    huggingface_config,
    sklearn_config,
    GlobalPreprocessConfig,
)
from model.base import BaseModel


def run_pipeline(preprocess_config: GlobalPreprocessConfig, models: List[BaseModel]):
    train_dataset, val_dataset, test_dataset = load_data(
        "data/original", preprocess_config
    )

    for model in models:
        model.fit(train_dataset, val_dataset)
        model.predict(test_dataset)


if __name__ == "__main__":
    run_pipeline(
        global_preprocess_config,
        [
            SKLearnModel(config=sklearn_config),
            HuggingfaceModel(config=huggingface_config),
        ],
    )
