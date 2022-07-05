from typing import List
from model.ensemble import Ensemble
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
from model.ensemble import Ensemble
from training.train import train_model

models = Ensemble(
    [
        SKLearnModel(config=sklearn_config),
        HuggingfaceModel(config=huggingface_config),
    ]
)


def run_pipeline(preprocess_config: GlobalPreprocessConfig, model: BaseModel = models):
    train_dataset, test_dataset = load_data("data/original", preprocess_config)

    model.preload()
    train_model(model, train_dataset)
    return model.predict(test_dataset)


if __name__ == "__main__":
    run_pipeline(global_preprocess_config, models)
