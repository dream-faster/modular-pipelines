from mopi.blocks.adaptors import ListOfListsToNumpy
from mopi.blocks.concat import ClassificationOutputConcat, DataSource
from mopi.blocks.ensemble import Ensemble
from mopi.blocks.models.random import AllOnesModel, RandomModel, AllZerosModel
from mopi.blocks.models.sklearn import SKLearnModel
from mopi.blocks.models.vader import VaderModel
from mopi.blocks.pipeline import Pipeline
from mopi.blocks.transformations import (
    SKLearnTransformation,
    SpacyTokenizer,
    TextStatisticTransformation,
)
from mopi.data.dataloader import MergedDataLoader, SamplingStrategy
from ..evaluation.classification import classification_metrics
from ..evaluation.calibration import calibration_metrics
from sklearn.preprocessing import MinMaxScaler
from mopi.type import (
    Experiment,
    DatasetSplit,
)
from mopi.utils.list import flatten
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

from .utils import populate_experiments_with_pipelines
from ...utils.dataset_analytics import print_dataset_analytics

### Models

tweet_eval_hate = DataSource("tweet_eval_hate", get_tweet_eval_dataloader("hate"))

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
    config=huggingface_config.set_attr("pretrained_model", "vinai/bertweet-base"),
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
    tweet_eval_hate,
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

ensemble_sklearn_vader = Ensemble(
    "ensemble_sklearn_vader", tweet_eval_hate, [sklearn, vader]
)

ensemble_sklearn_hf_vader = Ensemble(
    "ensemble_sklearn_hf_vader", tweet_eval_hate, [sklearn, vader, huggingface_baseline]
)

ensemble_sklearn_hf = Ensemble(
    "ensemble_sklearn_hf", tweet_eval_hate, [sklearn, huggingface_baseline]
)

ensemble_hf_vader = Ensemble(
    "ensemble_hf_vader",
    tweet_eval_hate,
    [huggingface_baseline, vader],
)


### Dataloaders

dataloader_tweeteval = get_tweet_eval_dataloader("hate")


data_merged_train = MergedDataLoader(
    [
        dataloader_tweeteval,
        get_tweets_hate_speech_detection_dataloader(),
        get_hate_speech_offensive_dataloader(),
        get_dynahate_dataloader(),
    ],
    sampling=SamplingStrategy.No,
)

data_merged_test = MergedDataLoader(
    [
        dataloader_tweeteval,
        get_tweets_hate_speech_detection_dataloader(),
        get_hate_speech_offensive_dataloader(),
        get_dynahate_dataloader(),
    ],
    sampling=SamplingStrategy.OverSample,
)

print_dataset_analytics(data_merged_train, DatasetSplit.train)
print_dataset_analytics(data_merged_train, DatasetSplit.test)


### Metrics

metrics = classification_metrics + calibration_metrics


### Tweeteval - tweeteval/hatecheck/dynahate/merged

cross_dataset_experiments_tweeteval = [
    Experiment(
        project_name="hate-speech-detection",
        run_name="tweeteval",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=dataloader_tweeteval,
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="tweeteval-tweeteval",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=dataloader_tweeteval,
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="tweeteval-hatecheck",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_hatecheck_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="tweeteval-dynahate",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_dynahate_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="tweeteval-offensive",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_hate_speech_offensive_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="tweeteval-merged",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=data_merged_test,
    ),
]


### Dynahate - tweeteval/hatecheck/dynahate/merged

cross_dataset_experiments_dynahate = [
    Experiment(
        project_name="hate-speech-detection",
        run_name="dynahate",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=get_dynahate_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="dynahate-tweeteval",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=dataloader_tweeteval,
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="dynahate-hatecheck",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_hatecheck_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="dynahate-dynahate",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_dynahate_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="dynahate-offensive",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_hate_speech_offensive_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="dynahate-merged",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=data_merged_test,
    ),
]

### Offensive - tweeteval/hatecheck/dynahate/merged

cross_dataset_experiments_offensive = [
    Experiment(
        project_name="hate-speech-detection",
        run_name="offensive",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=get_hate_speech_offensive_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="offensive-tweeteval",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=dataloader_tweeteval,
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="offensive-hatecheck",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_hatecheck_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="offensive-dynahate",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_dynahate_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="offensive-offensive",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_hate_speech_offensive_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="offensive-merged",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=data_merged_test,
    ),
]


### Merged - hatecheck/dynahate/merged

cross_dataset_experiments_merged = [
    Experiment(
        project_name="hate-speech-detection",
        run_name="merged",
        dataset_category=DatasetSplit.train,
        pipeline=sklearn,
        metrics=metrics,
        train=True,
        global_dataloader=data_merged_train,
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="merged-merged",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=data_merged_test,
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="merged-hatecheck",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_hatecheck_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="merged-dynahate",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_dynahate_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="merged-offensive",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=get_hate_speech_offensive_dataloader(),
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="merged-tweeteval",
        dataset_category=DatasetSplit.test,
        pipeline=sklearn,
        metrics=metrics,
        train=False,
        global_dataloader=dataloader_tweeteval,
    ),
]


pipelines_to_evaluate = [
    sklearn,
    # sklearn_calibrated,
    # sklearn_autocorrect,
    # sklearn_simple_nb,
    random,
    all_0s,
    all_1s,
    vader,
    # huggingface_baseline,
    # huggingface_hatebert,
    huggingface_bertweet,
    # text_statistics_pipeline,
    # ensemble_all,
    # ensemble_hf_vader,
    # ensemble_sklearn_hf,
    # ensemble_sklearn_vader,
    # meta_model_all,
]


all_tweeteval_crossexperiments = populate_experiments_with_pipelines(
    cross_dataset_experiments_tweeteval, pipelines_to_evaluate
)

all_dynahate_cross_experiments = populate_experiments_with_pipelines(
    cross_dataset_experiments_dynahate, pipelines_to_evaluate
)

all_offensive_cross_experiments = populate_experiments_with_pipelines(
    cross_dataset_experiments_dynahate, pipelines_to_evaluate
)

all_merged_cross_experiments = populate_experiments_with_pipelines(
    cross_dataset_experiments_merged, pipelines_to_evaluate
)
