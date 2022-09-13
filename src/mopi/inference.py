from mopi.run_dev import run_dev
from mopi.type import Experiment
from mopi.blocks.pipeline import Pipeline

from mopi.type import (
    Experiment,
    DatasetSplit,
)
from mopi.data.dataloader import PandasDataLoader
from mopi.type import PreprocessConfig
import pandas as pd
from mopi.library.evaluation.classification import classification_metrics
from mopi.blocks.io import load_pipeline
from mopi.constants import Const

from typing import Tuple, List


def get_inference_results(pipeline: Pipeline, texts: List[str]) -> Tuple[int, float]:

    text_with_fake_labels = [[text, 0] for text in texts] + [["dummy_text", 1]]

    dataloader = PandasDataLoader(
        "",
        PreprocessConfig(
            train_size=-1,
            val_size=-1,
            test_size=-1,
            input_col="text",
            label_col="label",
        ),
        pd.DataFrame([["", ""]], columns=["input", "label"]),
        pd.DataFrame(text_with_fake_labels, columns=["input", "label"]),
    )

    experiments_for_inference = [
        Experiment(
            project_name="hate-speech-detection-tweeteval",
            run_name="tweeteval",
            dataset_category=DatasetSplit.test,
            pipeline=pipeline,
            metrics=classification_metrics,
            train=False,
            global_dataloader=dataloader,
        ),
    ]
    successes = run_dev(experiments_for_inference, pure_inference=True)

    results = successes[0][1].get_all_predictions()[Const.final_output][: len(texts)]

    return results


def run_inference(model_name: str, tweets: List[str]) -> Tuple[int, float]:
    sk_learn_pipeline = load_pipeline(model_name)
    results = get_inference_results(sk_learn_pipeline, tweets)

    return results
