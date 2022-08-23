from copy import deepcopy
from typing import List

from blocks.adaptors import ListOfListsToNumpy
from blocks.concat import ClassificationOutputConcat, DataSource
from blocks.ensemble import Ensemble
from blocks.models.random import AllOnesModel, RandomModel, AllZerosModel
from blocks.models.sklearn import SKLearnModel
from blocks.models.vader import VaderModel
from blocks.pipeline import Pipeline
from blocks.transformations import (
    SKLearnTransformation,
    SpacyTokenizer,
    TextStatisticTransformation,
)
from data.dataloader import DataLoaderMerger
from ..evaluation.classification import classification_metrics
from ..evaluation.calibration import calibration_metrics
from sklearn.preprocessing import MinMaxScaler
from type import (
    Experiment,
    DatasetSplit,
)
from utils.list import flatten
from ..dataset.dynahate import get_dynahate_dataloader
from ..dataset.hatecheck import get_hatecheck_dataloader
from ..dataset.hatespeech_offensive import get_hate_speech_offensive_dataloader
from ..dataset.tweets_hate_speech_detection import (
    get_tweets_hate_speech_detection_dataloader,
)
from ..dataset.tweet_eval import get_tweet_eval_dataloader

from ..models.sklearn_voting import sklearn_config
from ..models.sklearn_simple import sklearn_config_simple_nb, sklearn_config_simple_lr
from ..models.huggingface import huggingface_config
from ..pipelines.huggingface import create_nlp_huggingface_pipeline
from ..pipelines.sklearn_nlp import create_nlp_sklearn_pipeline

from imblearn.over_sampling import RandomOverSampler

### Models

tweet_eval_hate = DataSource("tweet_eval_hate", get_tweet_eval_dataloader("hate"))
# input_data = DataSource("input")


### Pipelines


text_statistics_pipeline = Pipeline(
    "text_statistics",
    tweet_eval_hate,
    models=[
        SpacyTokenizer(),
        TextStatisticTransformation(),
        ListOfListsToNumpy(replace_nan=True),
        SKLearnTransformation(MinMaxScaler(feature_range=(0, 1), clip=True)),
        SKLearnModel("statistics_sklearn_ensemble", sklearn_config),
    ],
)

huggingface_baseline = create_nlp_huggingface_pipeline(
    title="hf-distillbert",
    input=tweet_eval_hate,
    config=huggingface_config,
    autocorrect=False,
)

huggingface_hatebert = create_nlp_huggingface_pipeline(
    title="hf-hatebert",
    input=tweet_eval_hate,
    config=huggingface_config.set_attr("pretrained_model", "GroNLP/hateBERT"),
    autocorrect=False,
)

huggingface_bertweet = create_nlp_huggingface_pipeline(
    title="hf-bertweet",
    input=tweet_eval_hate,
    config=huggingface_config.set_attr(
        "pretrained_model", "pysentimiento/bertweet-hate-speech"
    ),
    autocorrect=False,
)

sklearn = create_nlp_sklearn_pipeline(
    title="sklearn",
    input_data=tweet_eval_hate,
    sklearn_config=sklearn_config,
    autocorrect=False,
)

sklearn_calibrated = create_nlp_sklearn_pipeline(
    title="sklearn_calibrated",
    input_data=tweet_eval_hate,
    sklearn_config=sklearn_config.set_attr("calibrate", True),
    autocorrect=False,
)

sklearn_autocorrect = create_nlp_sklearn_pipeline(
    title="sklearn_autocorrect",
    input_data=tweet_eval_hate,
    sklearn_config=sklearn_config,
    autocorrect=True,
)

sklearn_simple_nb = create_nlp_sklearn_pipeline(
    title="sklearn_simple_nb",
    input_data=tweet_eval_hate,
    sklearn_config=sklearn_config_simple_nb,
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


### Dataloaders

dataloader_tweeteval = get_tweet_eval_dataloader("hate")


data_merged_train = DataLoaderMerger(
    [
        dataloader_tweeteval,
        get_tweets_hate_speech_detection_dataloader(),
        get_hate_speech_offensive_dataloader(),
        get_dynahate_dataloader(),
    ]
)


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

single_dataset_experiments_tweeteval_balanced = [
    Experiment(
        project_name="hate-speech-detection-tweeteval-balanced",
        run_name="tweeteval",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=dataloader_tweeteval.set_attr("sampler", RandomOverSampler()),
    ),
    Experiment(
        project_name="hate-speech-detection-tweeteval-balanced",
        run_name="tweeteval",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=dataloader_tweeteval.set_attr("sampler", RandomOverSampler()),
    ),
]

### Tweeteval - hatecheck/dynahate/merged

cross_dataset_experiments_tweeteval_hatecheck = [
    Experiment(
        project_name="hate-speech-detection-cross-tweeteval-hatecheck",
        run_name="tweeteval",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=dataloader_tweeteval,
    ),
    Experiment(
        project_name="hate-speech-detection-cross-tweeteval-hatecheck",
        run_name="hatecheck",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_hatecheck_dataloader(),
    ),
]


cross_dataset_experiments_tweeteval_dynahate = [
    Experiment(
        project_name="hate-speech-detection-cross-tweeteval-dynahate",
        run_name="tweeteval",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=dataloader_tweeteval,
    ),
    Experiment(
        project_name="hate-speech-detection-cross-tweeteval-dynahate",
        run_name="dynahate",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_dynahate_dataloader(),
    ),
]

cross_dataset_experiments_tweeteval_merged = [
    Experiment(
        project_name="hate-speech-detection-cross-tweeteval-merged",
        run_name="tweeteval",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=dataloader_tweeteval,
    ),
    Experiment(
        project_name="hate-speech-detection-cross-tweeteval-merged",
        run_name="merged",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=data_merged_train,
    ),
]

### Merged - hatecheck/dynahate/merged

cross_dataset_experiments_merged_hatecheck = [
    Experiment(
        project_name="hate-speech-detection-cross-merged-hatecheck",
        run_name="merged_dataset",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=data_merged_train,
    ),
    Experiment(
        project_name="hate-speech-detection-cross-merged-hatecheck",
        run_name="hatecheck",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_hatecheck_dataloader(),
    ),
]


cross_dataset_experiments_merged_dynahate = [
    Experiment(
        project_name="hate-speech-detection-cross-merged-dynahate",
        run_name="merged_dataset",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=data_merged_train,
    ),
    Experiment(
        project_name="hate-speech-detection-cross-merged-dynahate",
        run_name="dynahate",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_dynahate_dataloader(),
    ),
]


cross_dataset_experiments_merged_merged = [
    Experiment(
        project_name="hate-speech-detection-cross-merged-merged",
        run_name="merged_dataset",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=data_merged_train,
    ),
    Experiment(
        project_name="hate-speech-detection-cross-merged-merged",
        run_name="merged_dataset",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=data_merged_train,
    ),
]

pipelines_to_evaluate = [
    sklearn,
    sklearn_calibrated,
    sklearn_autocorrect,
    sklearn_simple_nb,
    random,
    all_0s,
    all_1s,
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


all_tweeteval_experiments = populate_experiments_with_pipelines(
    single_dataset_experiments_tweeteval, pipelines_to_evaluate
) + populate_experiments_with_pipelines(
    single_dataset_experiments_tweeteval_balanced, pipelines_to_evaluate
)

all_tweeteval_cross_experiments = (
    populate_experiments_with_pipelines(
        cross_dataset_experiments_tweeteval_hatecheck, pipelines_to_evaluate
    )
    + populate_experiments_with_pipelines(
        cross_dataset_experiments_tweeteval_dynahate, pipelines_to_evaluate
    )
    + populate_experiments_with_pipelines(
        cross_dataset_experiments_tweeteval_merged, pipelines_to_evaluate
    )
)

all_merged_cross_experiments = (
    populate_experiments_with_pipelines(
        cross_dataset_experiments_merged_hatecheck, pipelines_to_evaluate
    )
    + populate_experiments_with_pipelines(
        cross_dataset_experiments_merged_dynahate, pipelines_to_evaluate
    )
    + populate_experiments_with_pipelines(
        cross_dataset_experiments_merged_merged, pipelines_to_evaluate
    )
)
