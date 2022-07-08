from typing import Callable

from configs.config import HuggingfaceConfig
from datasets import Dataset
import numpy as np
from configs.constants import Const
import pandas as pd

def run_inference_pipeline(
    model: Callable, test_data: Dataset, config: HuggingfaceConfig
) -> pd.DataFrame:

    predictions = model(test_data[Const.input_col], top_k=config.num_classes)
    probs = [
        tuple([label_score["score"] for label_score in prediction])
        for prediction in predictions
    ]
    predicitions = [np.argmax(prob) for prob in probs]

    return pd.DataFrame({Const.preds_col: predicitions, Const.probs_col: probs})
