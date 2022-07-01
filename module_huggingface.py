from typing import Callable
from transformers import pipeline
from config import HuggingfaceConfig, huggingface_config
import pandas as pd
from data.dataloader import load_data
from datasets import Dataset


def load_module(module_name: str) -> Callable:
    return pipeline(task="sentiment-analysis", model=module_name)


def run_inference(config: HuggingfaceConfig, dataset: Dataset) -> list:
    sentiment_model = load_module(config.user_name + "/" + config.repo_name)

    predictions = sentiment_model.predict(dataset["text"])

    return predictions


def run_inference_pipeline(config: HuggingfaceConfig) -> None:
    _, _, test_data = load_data(config.data_from_huggingface)
    predictions = run_inference(huggingface_config, test_data[: config.test_size])

    print(predictions)


if __name__ == "__main__":
    run_inference_pipeline(huggingface_config)
