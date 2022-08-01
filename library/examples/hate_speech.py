from blocks.adaptors import ListOfListsToNumpy
from blocks.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from blocks.data import DataSource
from blocks.ensemble import Ensemble
from blocks.models.huggingface import HuggingfaceModel
from blocks.models.random import RandomModel
from blocks.models.sklearn import SKLearnModel
from blocks.models.vader import VaderModel
from blocks.pipeline import Pipeline
from blocks.transformations import (
    Lemmatizer,
    SKLearnTransformation,
    SpacyTokenizer,
    TextStatisticTransformation,
)
from data.merge import merge_datasets
from data.transformation import transform_dataset
from data.transformation_hatecheck import transform_hatecheck_dataset
from data.transformation_hatespeech_detection import (
    transform_hatespeech_detection_dataset,
)
from data.transformation_hatespeech_offensive import (
    transform_hatespeech_offensive_dataset,
)
from datasets.load import load_dataset
from library.evaluation import calibration_metrics, classification_metrics
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from transformers.training_args import TrainingArguments
from type import (
    HFTaskTypes,
    HuggingfaceConfig,
    LoadOrigin,
    PreprocessConfig,
    Experiment,
    SKLearnConfig,
)
from utils.flatten import remove_none, flatten


preprocess_config = PreprocessConfig(
    train_size=-1,
    val_size=-1,
    test_size=-1,
    input_col="text",
    label_col="label",
)

### Models

huggingface_config = HuggingfaceConfig(
    preferred_load_origin=LoadOrigin.local,
    pretrained_model="distilbert-base-uncased",
    task_type=HFTaskTypes.sentiment_analysis,
    user_name="semy",
    save_remote=True,
    save=True,
    num_classes=2,
    val_size=0.1,
    force_fit=False,
    remote_name_override=None,
    training_args=TrainingArguments(
        output_dir="",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=True,
        log_level="critical",
        report_to="none",
        optim="adamw_torch",
        logging_strategy="steps",
        evaluation_strategy="epoch",
        logging_steps=1,
        # eval_steps = 10
    ),
)

sklearn_config = SKLearnConfig(
    force_fit=False,
    save=True,
    preferred_load_origin=LoadOrigin.local,
    classifier=VotingClassifier(
        estimators=[
            ("nb", MultinomialNB()),
            ("lg", LogisticRegression()),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100, max_depth=20, random_state=0
                ),
            ),
        ],
        voting="soft",
    ),
    one_vs_rest=False,
    save_remote=False,
)

sklearn_config_simple = SKLearnConfig(
    preferred_load_origin=LoadOrigin.local,
    force_fit=False,
    save=True,
    classifier=MultinomialNB(),
    one_vs_rest=False,
    save_remote=False,
)


input_data = DataSource("input")


### Pipelines


def create_nlp_sklearn_pipeline(autocorrect: bool, simple: bool = False) -> Pipeline:
    return Pipeline(
        "nlp_sklearn_autocorrect" if autocorrect else "nlp_sklearn",
        input_data,
        remove_none(
            [
                SpellAutocorrectAugmenter(fast=True) if autocorrect else None,
                SpacyTokenizer(),
                Lemmatizer(remove_stopwords=False),
                SKLearnTransformation(
                    TfidfVectorizer(
                        max_features=100000,
                        ngram_range=(1, 3),
                    )
                ),
                SKLearnModel(
                    "nlp-sklearn", sklearn_config if simple else sklearn_config_simple
                ),
            ]
        ),
    )


def create_nlp_huggingface_pipeline(autocorrect: bool) -> Pipeline:
    return Pipeline(
        "nlp_hf_autocorrect" if autocorrect else "nlp_hf",
        input_data,
        remove_none(
            [
                SpellAutocorrectAugmenter(fast=True) if autocorrect else None,
                HuggingfaceModel("hf-model", huggingface_config),
            ]
        ),
    )


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

huggingface_baseline = create_nlp_huggingface_pipeline(autocorrect=False)
sklearn = create_nlp_sklearn_pipeline(autocorrect=False)
sklearn_autocorrect = create_nlp_sklearn_pipeline(autocorrect=True)

sklearn_simple = create_nlp_sklearn_pipeline(autocorrect=False)
random = Pipeline("random", input_data, [RandomModel("random")])
vader = Pipeline("vader", input_data, [VaderModel("vader")])

ensemble_all = Ensemble(
    "ensemble_all-all",
    [sklearn, huggingface_baseline, text_statistics_pipeline, vader],
)

ensemble_sklearn_vader = Ensemble("ensemble_sklearn_vader", [sklearn, vader])

ensemble_sklearn_hf = Ensemble("ensemble_sklearn_hf", [sklearn, huggingface_baseline])

ensemble_hf_vader = Ensemble(
    "ensemble_hf_vader",
    [huggingface_baseline],
)


### Datasets

data_tweet_eval_hate_speech = transform_dataset(
    load_dataset("tweet_eval", "hate"), preprocess_config
)

data_tweets_hate_speech_detection = transform_hatespeech_detection_dataset(
    load_dataset("tweets_hate_speech_detection"),
    config=PreprocessConfig(
        train_size=-1,
        val_size=-1,
        test_size=-1,
        input_col="tweet",
        label_col="label",
    ),
)
data_hatecheck = transform_hatecheck_dataset(
    load_dataset("Paul/hatecheck"), preprocess_config
)

data_hate_speech_offensive = transform_hatespeech_offensive_dataset(
    load_dataset("hate_speech_offensive"),
    config=PreprocessConfig(
        train_size=-1,
        val_size=-1,
        test_size=-1,
        input_col="tweet",
        label_col="class",
    ),
)


data_merged_train = merge_datasets(
    [
        data_tweet_eval_hate_speech[0],
        data_tweets_hate_speech_detection[0],
        data_hate_speech_offensive[0],
    ]
)

### Metrics

metrics = classification_metrics + calibration_metrics


### Run Configs

tweeteval_hate_speech_experiments = [
    Experiment(
        project_name="hate-speech-detection",
        run_name="tweeteval",
        dataset=data_tweet_eval_hate_speech[0],
        pipeline=sklearn,
        preprocessing_config=preprocess_config,
        metrics=metrics,
        train=True,
    ),
    Experiment(
        project_name="hate-speech-detection",
        run_name="tweeteval",
        dataset=data_tweet_eval_hate_speech[1],
        pipeline=sklearn,
        preprocessing_config=preprocess_config,
        metrics=metrics,
        train=False,
    ),
]


cross_dataset_experiments = [
    Experiment(
        project_name="hate-speech-detection-merged",
        run_name="merged_dataset",
        dataset=data_merged_train,
        pipeline=sklearn,
        preprocessing_config=preprocess_config,
        metrics=metrics,
        train=True,
    ),
    Experiment(
        project_name="hate-speech-detection-cross-val",
        run_name="hatecheck",
        dataset=data_hatecheck[1],
        pipeline=sklearn,
        preprocessing_config=preprocess_config,
        metrics=metrics,
        train=False,
    ),
]

pipelines_to_evaluate = [
    sklearn,
    sklearn_autocorrect,
    random,
    vader,
    huggingface_baseline,
    ensemble_all,
    ensemble_hf_vader,
    ensemble_sklearn_hf,
    ensemble_sklearn_vader,
]


def set_pipeline(experiment: Experiment, pipeline: Pipeline) -> Experiment:
    experiment.pipeline = pipeline
    return experiment


all_cross_dataset_experiments = flatten(
    [
        [
            [
                set_pipeline(experiment, pipeline)
                for experiment in cross_dataset_experiments
            ]
        ]
        for pipeline in pipelines_to_evaluate
    ]
)
