from typing import List
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
from adapter.concatenator import ConcatenatorAdapter
from adapter.identity import IdentityAdapter


def run_pipeline(preprocess_config: GlobalPreprocessConfig):

    pipeline = Linear(
        [
            IdentityAdapter("global"),
            SKLearnModel(name="sklearn_1", config=sklearn_config, train=True),
            ConcatenatorAdapter(name="concatenate_1", keys=["global", "sklearn_1"]),
            SKLearnModel(name="sklearn_2", config=sklearn_config, train=True),
            ConcatenatorAdapter(name="concatenate_2", keys=["global", "sklearn_2"]),
            SKLearnModel(name="sklearn_3", config=sklearn_config, train=True),
            ConcatenatorAdapter(
                name="concatenate_3", keys=["global", "concatenate_1", "concatenate_2"]
            ),
            SKLearnModel(name="sklearn_4", config=sklearn_config, train=True),
            # HuggingfaceModel(name='huggingface_1', config=huggingface_config, train=False),
        ]
    )

    train_dataset, test_dataset = load_data("data/original", preprocess_config)

    pipeline.preload()
    pipeline.train(train_dataset)
    return pipeline.predict(test_dataset)


if __name__ == "__main__":
    run_pipeline(global_preprocess_config)
