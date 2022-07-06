from typing import List
from model.sequential import Sequential
from model.huggingface import HuggingfaceModel

from model.sklearn import SKLearnModel
from data.dataloader import load_data
from configs.config import (
    global_preprocess_config,
    huggingface_config,
    sklearn_config,
    GlobalPreprocessConfig,
)
from model.base import BaseModel
from model.linear import Linear
from model.augmenters.identity import IdentityAugmenter


def run_pipeline(preprocess_config: GlobalPreprocessConfig):

    pipeline = Linear(
        [
            IdentityAugmenter(),
            SKLearnModel(name="sklearn_1", config=sklearn_config, train=False),
            # HuggingfaceModel(name='huggingface_1', config=huggingface_config, train=False),
        ]
    )

    train_dataset, test_dataset = load_data("data/original", preprocess_config)

    pipeline.preload()
    pipeline.train(train_dataset)
    return pipeline.predict(test_dataset)


if __name__ == "__main__":
    run_pipeline(global_preprocess_config)
