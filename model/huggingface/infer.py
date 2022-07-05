from typing import Callable, Tuple, List

from configs.config import HuggingfaceConfig
from datasets import Dataset
import numpy as np
from type import Label, Probabilities
from configs.constants import DataConst


def run_inference_pipeline(
    model: Callable, test_data: Dataset, config: HuggingfaceConfig
) -> List[Tuple[Label, Probabilities]]:

    predictions = model(test_data[DataConst.input_name], top_k=config.num_classes)
    scores = [
        [label_score["score"] for label_score in prediction]
        for prediction in predictions
    ]
    labels = [np.argmax(score) for score in scores]

    return list(zip(labels, scores))
