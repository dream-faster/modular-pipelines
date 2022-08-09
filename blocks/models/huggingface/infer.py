from typing import Callable, List, Tuple

import numpy as np
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines import pipeline
from configs.constants import Const
from type import HuggingfaceConfig, PredsWithProbs


def run_inference(
    model: PreTrainedModel,
    test_data: Dataset,
    config: HuggingfaceConfig,
    tokenizer: PreTrainedTokenizer,
    device,
) -> List[PredsWithProbs]:

    inference_pipeline = pipeline(
        task=config.task_type.value,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    scores = inference_pipeline(test_data[Const.input_col], top_k=config.num_classes)
    probs = [convert_scores_dict_to_probs(score) for score in scores]
    predicitions = [np.argmax(prob) for prob in probs]

    return list(zip(predicitions, probs))


def take_first(elem):
    return elem[0]


def convert_scores_dict_to_probs(scores: List[dict]) -> List[Tuple]:
    sorted_scores = sorted(
        [
            (int(scores_dict["label"].split("_")[1]), scores_dict["score"])
            for scores_dict in scores
        ],
        key=take_first,
    )
    return [item[1] for item in sorted_scores]
