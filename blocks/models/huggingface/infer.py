from typing import Callable, List
from type import HuggingfaceConfig, PredsWithProbs
from datasets import Dataset
import numpy as np
from configs.constants import Const
import pandas as pd


def run_inference_pipeline(
    model: Callable, test_data: Dataset, config: HuggingfaceConfig
) -> List[PredsWithProbs]:

    predictions = model(test_data[Const.input_col], top_k=config.num_classes)
    probs = [
        tuple([label_score["score"] for label_score in prediction])
        for prediction in predictions
    ]
    predicitions = [np.argmax(prob) for prob in probs]

    return list(zip(predicitions, probs))
