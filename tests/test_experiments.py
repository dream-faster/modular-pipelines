from ast import Store
from copy import deepcopy
import pandas as pd
from typing import List
from src.mopi.blocks.base import DataSource

from src.mopi.blocks.concat import ClassificationOutputConcat
from src.mopi.blocks.ensemble import Ensemble
from src.mopi.blocks.models.random import AllOnesModel, RandomModel, AllZerosModel
from src.mopi.blocks.models.sklearn import SKLearnModel
from src.mopi.blocks.pipeline import Pipeline
from src.mopi.constants import Const
from src.mopi.library.evaluation.classification import classification_metrics
from src.mopi.library.evaluation.calibration import calibration_metrics
from src.mopi.type import (
    Experiment,
    DatasetSplit,
    StagingConfig,
    StagingNames,
)
from src.mopi.utils.list import flatten
from src.mopi.library.models.sklearn_voting import sklearn_config
from src.mopi.library.models.huggingface import huggingface_config
from src.mopi.library.pipelines.huggingface import create_nlp_huggingface_pipeline
from src.mopi.library.pipelines.sklearn_nlp import create_nlp_sklearn_pipeline
from src.mopi.library.experiments.utils import populate_experiments_with_pipelines
from src.mopi.library.dataset.tweet_eval import get_tweet_eval_dataloader
from src.mopi.blocks.models.vader import VaderModel

from src.mopi.library.models.sklearn_simple import (
    sklearn_config_simple_nb,
    sklearn_config_simple_lr,
)
from src.mopi.run import run


def create_experiments() -> List[Experiment]:
    ### Datasources

    tweet_eval_hate = DataSource(
        "tweet_eval_hate", get_tweet_eval_dataloader("hate", shuffle_first=True)
    )

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

    dataloader_tweeteval = get_tweet_eval_dataloader("hate", shuffle_first=True)

    ### Metrics

    metrics = classification_metrics + calibration_metrics

    ### Single train/test split

    single_dataset_experiments_tweeteval = [
        Experiment(
            project_name="test",
            run_name="tweeteval",
            dataset_category=DatasetSplit.train,
            pipeline=sklearn,
            metrics=metrics,
            train=True,
            global_dataloader=dataloader_tweeteval,
        ),
        Experiment(
            project_name="test",
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


def __check_correct_stats(stats: pd.Series, experiment: Experiment):
    correct_ranges = [
        ("f1_binary", 0.0, 1.0),
        ("accuracy", 0.0, 1.0),
        ("precision_binary", 0.0, 1.0),
        ("recall_binary", 0.0, 1.0),
        ("roc_auc", 0.0, 1.0),
        ("mce", 0.0, 1.0),
    ]

    context_string = f"~ Failed for {experiment.run_name} with stats: {stats}."
    for metric, minimum, maximum in correct_ranges:
        assert (
            stats[metric] >= minimum
        ), f"{metric}: {stats[metric]} is smaller than {minimum} (minimum) {context_string}"
        assert (
            stats[metric] <= maximum
        ), f"{metric}: {stats[metric]} is larger than {maximum} (maximum) {context_string}"


def test_experiments():
    dev_config = StagingConfig(
        name=StagingNames.dev,
        save_remote=False,
        log_remote=False,
        limit_dataset_to=100,
    )
    dev_config_logging = StagingConfig(
        name=StagingNames.dev,
        save_remote=False,
        log_remote=True,
        limit_dataset_to=100,
    )

    experiments = create_experiments()

    for i, experiment in enumerate(experiments):

        if i == 0:
            config = dev_config_logging
        else:
            config = dev_config

        successes = run(
            [experiment],
            staging_config=config,
        )

        for experiment, store in successes:
            stats = store.get_all_stats()[Const.final_eval_name]
            __check_correct_stats(stats, experiment)
