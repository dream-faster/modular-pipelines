from dataclasses import dataclass
from typing import Callable, Tuple
from transformers import pipeline
from config import HuggingfaceConfig, huggingface_config
from datasets import Dataset


def load_module(module_name: str) -> Callable:
    return pipeline(task="sentiment-analysis", model=module_name)


def run_inference_pipeline(
    test_data: Dataset, config: HuggingfaceConfig
) -> Tuple[list, list]:
    sentiment_model = load_module(config.user_name + "/" + config.repo_name)

    predictions = sentiment_model(test_data["text"])
    labels = [int(prediction["label"].split("_")[1]) for prediction in predictions]
    scores = [prediction["score"] for prediction in predictions]

    return labels, scores
