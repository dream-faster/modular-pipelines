from typing import Callable, Tuple, List, Union

from config import HuggingfaceConfig
from datasets import Dataset
import numpy as np
from type import Label, Probabilities
from transformers import Trainer


def run_inference_pipeline(
    model: Union[Callable, Trainer], test_data: Dataset, config: HuggingfaceConfig
) -> List[Tuple[Label, Probabilities]]:

    predictions = model(test_data["text"], top_k=config.num_classes)
    scores = [
        [label_score["score"] for label_score in prediction]
        for prediction in predictions
    ]
    labels = [np.argmax(score) for score in scores]

    return list(zip(labels, scores))
