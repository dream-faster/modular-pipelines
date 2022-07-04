from dataclasses import dataclass
from typing import Callable, Tuple, List
from transformers import pipeline
from config import HuggingfaceConfig, huggingface_config
from datasets import Dataset
import numpy as np
from type import Label, Probabilities


def load_module(module_name: str) -> Callable:
    return pipeline(task="sentiment-analysis", model=module_name)


def run_inference_pipeline(
    test_data: Dataset, config: HuggingfaceConfig
) -> List[Tuple[Label, Probabilities]]:
    sentiment_model = load_module(config.user_name + "/" + config.repo_name)

    predictions = sentiment_model(test_data["text"], return_all_scores=True)
    scores = [
        [label_score["score"] for label_score in prediction]
        for prediction in predictions
    ]
    labels = [np.argmax(score) for score in scores]

    return list(zip(labels, scores))
