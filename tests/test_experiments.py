from copy import deepcopy
from typing import List

from blocks.concat import ClassificationOutputConcat
from blocks.ensemble import Ensemble
from blocks.models.random import AllOnesModel, RandomModel, AllZerosModel
from blocks.models.sklearn import SKLearnModel
from blocks.pipeline import Pipeline
from library.evaluation.classification import classification_metrics
from library.evaluation.calibration import calibration_metrics
from type import (
    Experiment,
    DatasetSplit,
)
from utils.list import flatten
from library.models.sklearn_voting import sklearn_config
from library.models.huggingface import huggingface_config
from library.pipelines.huggingface import create_nlp_huggingface_pipeline
from library.pipelines.sklearn_nlp import create_nlp_sklearn_pipeline


### Pipelines

huggingface_baseline = create_nlp_huggingface_pipeline(
    title="hf-distillbert",
    input=tweet_eval_hate,
    config=huggingface_config,
    autocorrect=False,
)


sklearn = create_nlp_sklearn_pipeline(
    title="sklearn",
    input_data=tweet_eval_hate,
    sklearn_config=sklearn_config,
    autocorrect=False,
)


random = Pipeline("random", tweet_eval_hate, [RandomModel("random")])
all_0s = Pipeline("all_0s", tweet_eval_hate, [AllZerosModel("all_0s")])
all_1s = Pipeline("all_1s", tweet_eval_hate, [AllOnesModel("all_1s")])

vader = Pipeline("vader", tweet_eval_hate, [VaderModel("vader")])

ensemble_all = Ensemble(
    "ensemble_all-all",
    [sklearn, huggingface_baseline, text_statistics_pipeline, vader],
)

meta_model_all = Pipeline(
    "meta_model_all",
    ClassificationOutputConcat(
        "all_models",
        [sklearn, huggingface_baseline, text_statistics_pipeline, vader],
        datasource_labels=tweet_eval_hate,
    ),
    [SKLearnModel("meta_model", sklearn_config_simple_lr.set_attr("calibrate", True))],
)

ensemble_sklearn_vader = Ensemble("ensemble_sklearn_vader", [sklearn, vader])

ensemble_sklearn_hf_vader = Ensemble(
    "ensemble_sklearn_hf_vader", [sklearn, vader, huggingface_baseline]
)

ensemble_sklearn_hf = Ensemble("ensemble_sklearn_hf", [sklearn, huggingface_baseline])

ensemble_hf_vader = Ensemble(
    "ensemble_hf_vader",
    [huggingface_baseline],
)


dataloader_tweeteval = get_tweet_eval_dataloader("hate")

### Metrics

metrics = classification_metrics + calibration_metrics


### Single train/test split

single_dataset_experiments_tweeteval = [
    Experiment(
        project_name="hate-speech-detection-tweeteval",
        run_name="tweeteval",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=dataloader_tweeteval,
    ),
    Experiment(
        project_name="hate-speech-detection-tweeteval",
        run_name="tweeteval",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=dataloader_tweeteval,
    ),
]


pipelines_to_evaluate = [
    sklearn,
    random,
    all_0s,
    all_1s,
    huggingface_baseline,
    ensemble_all,
    meta_model_all,
]


all_tweeteval_experiments = populate_experiments_with_pipelines(
    single_dataset_experiments_tweeteval, pipelines_to_evaluate
)
