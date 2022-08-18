from copy import deepcopy
from typing import List

from blocks.adaptors import ListOfListsToNumpy
from blocks.concat import ClassificationOutputConcat, DataSource
from blocks.ensemble import Ensemble
from blocks.models.random import RandomModel
from blocks.models.sklearn import SKLearnModel
from blocks.models.vader import VaderModel
from blocks.pipeline import Pipeline
from blocks.transformations import (
    SKLearnTransformation,
    SpacyTokenizer,
    TextStatisticTransformation,
)
from data.dataloader import DataLoaderMerger
from library.evaluation.classification import classification_metrics
from library.evaluation.calibration import calibration_metrics
from sklearn.preprocessing import MinMaxScaler
from type import (
    Experiment,
    DatasetSplit,
)
from utils.flatten import flatten
from ...utils.setter import clone_and_set
from ..dataset.hatecheck import get_hatecheck_dataloader
from ..dataset.hatespeech_offensive import get_hate_speech_offensive_dataloader
from ..dataset.tweets_hate_speech_detection import (
    get_tweets_hate_speech_detection_dataloader,
)
from ..dataset.tweet_eval import get_tweet_eval_dataloader

from ..models.sklearn_voting import sklearn_config
from ..models.sklearn_simple import sklearn_config_simple
from ..models.huggingface import huggingface_config
from ..pipelines.huggingface import create_nlp_huggingface_pipeline
from ..pipelines.sklearn_nlp import create_nlp_sklearn_pipeline


### Models


input_data = DataSource("input")


### Pipelines


text_statistics_pipeline = Pipeline(
    "text_statistics",
    input_data,
    models=[
        SpacyTokenizer(),
        TextStatisticTransformation(),
        ListOfListsToNumpy(replace_nan=True),
        SKLearnTransformation(MinMaxScaler(feature_range=(0, 1), clip=True)),
        SKLearnModel("statistics_sklearn_ensemble", sklearn_config),
    ],
)

huggingface_baseline = create_nlp_huggingface_pipeline(
    input=input_data, config=huggingface_config, autocorrect=False
)

huggingface_hatebert = create_nlp_huggingface_pipeline(
    input=input_data,
    config=clone_and_set(
        huggingface_config,
        {"id": "huggingface_hatebert", "pretrained_model": "GroNLP/hateBERT"},
    ),
    autocorrect=False,
)

huggingface_bertweet = create_nlp_huggingface_pipeline(
    input=input_data,
    config=clone_and_set(
        huggingface_config,
        {
            "id": "huggingface_bertweet",
            "pretrained_model": "pysentimiento/bertweet-hate-speech",
        },
    ),
    autocorrect=False,
)

sklearn = create_nlp_sklearn_pipeline(
    title="sklearn",
    input_data=input_data,
    sklearn_config=sklearn_config,
    autocorrect=False,
)
sklearn_autocorrect = create_nlp_sklearn_pipeline(
    title="sklearn_autocorrect",
    input_data=input_data,
    sklearn_config=sklearn_config,
    autocorrect=True,
)

sklearn_simple = create_nlp_sklearn_pipeline(
    title="sklearn_simple",
    input_data=input_data,
    sklearn_config=sklearn_config_simple,
    autocorrect=False,
)
random = Pipeline("random", input_data, [RandomModel("random")])
vader = Pipeline("vader", input_data, [VaderModel("vader")])

ensemble_all = Ensemble(
    "ensemble_all-all",
    [sklearn, huggingface_baseline, text_statistics_pipeline, vader],
)

meta_model_all = Pipeline(
    "meta_model_all",
    ClassificationOutputConcat(
        "all_models", [sklearn, huggingface_baseline, text_statistics_pipeline, vader]
    ),
    [SKLearnModel("meta_model", sklearn_config)],
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


### Datasets


data_merged_train = DataLoaderMerger(
    [
        get_tweet_eval_dataloader("hate"),
        get_tweets_hate_speech_detection_dataloader(),
        get_hate_speech_offensive_dataloader(),
    ]
)


### Metrics

metrics = classification_metrics + calibration_metrics


### Run Configs

tweeteval_hate_speech_experiments = [
    Experiment(
        project_name="hate-speech-detection",
        run_name="tweeteval",
        dataloader=get_tweet_eval_dataloader("hate"),
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="tweeteval",
        dataloader=get_tweet_eval_dataloader("hate"),
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
    ),
]


cross_dataset_experiments = [
    Experiment(
        project_name="hate-speech-detection-cross-val",
        run_name="merged_dataset",
        dataloader=data_merged_train,
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
    ),
    Experiment(
        project_name="hate-speech-detection-cross-val",
        run_name="hatecheck",
        dataloader=get_hatecheck_dataloader(),
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
    ),
]

pipelines_to_evaluate = [
    sklearn,
    sklearn_autocorrect,
    sklearn_simple,
    random,
    vader,
    huggingface_baseline,
    huggingface_hatebert,
    huggingface_bertweet,
    text_statistics_pipeline,
    ensemble_all,
    ensemble_hf_vader,
    ensemble_sklearn_hf,
    ensemble_sklearn_vader,
    meta_model_all,
]


def set_pipeline(experiment: Experiment, pipeline: Pipeline) -> Experiment:
    experiment.pipeline = pipeline
    return experiment


def populate_experiments_with_pipelines(
    experiments: List[Experiment], pipelines: List[Pipeline]
) -> List[Experiment]:
    return flatten(
        [
            [
                [
                    set_pipeline(deepcopy(experiment), pipeline)
                    for experiment in experiments
                ]
            ]
            for pipeline in pipelines
        ]
    )


all_cross_dataset_experiments = populate_experiments_with_pipelines(
    cross_dataset_experiments, pipelines_to_evaluate
)
all_tweeteval_hate_speech_experiments = populate_experiments_with_pipelines(
    tweeteval_hate_speech_experiments, pipelines_to_evaluate
)
