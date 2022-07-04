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

    predictions = []
    for model in models:
        if not model.is_fitted() or model.config.force_fit:
            model.fit(train_dataset, val_dataset)
        predictions.append(model.predict(test_dataset))

    print(predictions)


if __name__ == "__main__":
    run_pipeline(
        global_preprocess_config,
        [
            SKLearnModel(config=sklearn_config),
            HuggingfaceModel(config=huggingface_config),
        ],
    )
