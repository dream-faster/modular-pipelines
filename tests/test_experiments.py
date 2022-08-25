from ast import Store
from copy import deepcopy
import pandas as pd
from typing import List
from blocks.base import DataSource

from blocks.concat import ClassificationOutputConcat
from blocks.ensemble import Ensemble
from blocks.models.random import AllOnesModel, RandomModel, AllZerosModel
from blocks.models.sklearn import SKLearnModel
from blocks.pipeline import Pipeline
from constants import Const
from library.evaluation.classification import classification_metrics
from library.evaluation.calibration import calibration_metrics
from type import (
    Experiment,
    DatasetSplit,
    StagingConfig,
    StagingNames,
)
from utils.list import flatten
from library.models.sklearn_voting import sklearn_config
from library.models.huggingface import huggingface_config
from library.pipelines.huggingface import create_nlp_huggingface_pipeline
from library.pipelines.sklearn_nlp import create_nlp_sklearn_pipeline
from library.experiments.utils import populate_experiments_with_pipelines
from library.dataset.tweet_eval import get_tweet_eval_dataloader
from blocks.models.vader import VaderModel

from library.models.sklearn_simple import (
    sklearn_config_simple_nb,
    sklearn_config_simple_lr,
)


def create_experiments() -> List[Experiment]:
    ### Datasources

    tweet_eval_hate = DataSource("tweet_eval_hate", get_tweet_eval_dataloader("hate"))

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
        tweet_eval_hate,
        [sklearn, huggingface_baseline, vader],
    )

    meta_model_all = Pipeline(
        "meta_model_all",
        ClassificationOutputConcat(
            "all_models",
            [sklearn, huggingface_baseline, vader],
            datasource_labels=tweet_eval_hate,
        ),
        [
            SKLearnModel(
                "meta_model", sklearn_config_simple_lr.set_attr("calibrate", True)
            )
        ],
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
        random,
        all_0s,
        all_1s,
        sklearn,
        huggingface_baseline,
        ensemble_all,
        meta_model_all,
    ]

    all_tweeteval_experiments = populate_experiments_with_pipelines(
        single_dataset_experiments_tweeteval, pipelines_to_evaluate
    )

    return all_tweeteval_experiments


from run import run


def __check_correct_stats(stats: pd.Series):
    correct_ranges = [
        ("f1", 0.3, 1.0),
        ("accuracy", 0.3, 1.0),
        ("precision", 0.3, 1.0),
        ("recall", 0.3, 1.0),
        ("roc_auc", 0.1, 1.0),
        ("mce", 0.3, 1.0),
    ]

    return any(
        [
            True if stats[metric] > minimum and stats[metric] < maximum else False
            for metric, minimum, maximum in correct_ranges
        ]
    )


def test_experiments():
    dev_config = StagingConfig(
        name=StagingNames.dev,
        save_remote=False,
        log_remote=False,
        limit_dataset_to=100,
    )

    successes = run(
        create_experiments(),
        staging_config=dev_config,
    )

    for experiment, store in successes:
        stats = store.get_all_stats()[Const.final_eval_name]
        assert __check_correct_stats(
            stats
        ), f"Failed for {experiment.run_name} with stats:{stats}"
