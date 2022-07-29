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
from data.transformation import transform_dataset
from data.transformation_hatecheck import transform_hatecheck_dataset
from datasets.load import load_dataset
from library.evaluation import classification_metrics
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from transformers import TrainingArguments
from type import (
    HuggingfaceConfig,
    LoadOrigin,
    PreprocessConfig,
    RunConfig,
    SKLearnConfig,
)
from utils.flatten import remove_none

preprocess_config = PreprocessConfig(
    train_size=-1,
    val_size=-1,
    test_size=-1,
    input_col="text",
    label_col="label",
)

huggingface_config = HuggingfaceConfig(
    preferred_load_origin=LoadOrigin.remote,
    pretrained_model="distilbert-base-uncased",
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
nlp_sklearn = create_nlp_sklearn_pipeline(autocorrect=False)
nlp_sklearn_autocorrect = create_nlp_sklearn_pipeline(autocorrect=True)

nlp_sklearn_simple = create_nlp_sklearn_pipeline(autocorrect=False)
random = Pipeline("random", input_data, [RandomModel("random")])
vader = Pipeline("vader", input_data, [VaderModel("vader")])

ensemble_pipeline = Ensemble(
    "ensemble", [nlp_sklearn, nlp_sklearn_autocorrect, text_statistics_pipeline]
)

ensemble_pipeline_hf = Ensemble(
    "ensemble_hf_sklearn", [nlp_sklearn, huggingface_baseline]
)

ensemble_pipeline_hf_statistic = Ensemble(
    "ensemble_hf_statistic", [text_statistics_pipeline, huggingface_baseline]
)

ensemble_pipeline_hf_statistic_sklearn = Ensemble(
    "ensemble_hf_statistic_sklearn",
    [nlp_sklearn, text_statistics_pipeline, huggingface_baseline],
)


data_emoji = transform_dataset(load_dataset("tweet_eval", "emoji"), preprocess_config)

data_tweet_eval_hate_speech = transform_dataset(
    load_dataset("tweet_eval", "hate"), preprocess_config
)
data_hatecheck = transform_hatecheck_dataset(
    load_dataset("Paul/hatecheck"), preprocess_config
)

tweeteval_hate_speech_run_configs = [
    RunConfig(
        run_name="hate-speech-detection",
        dataset=data_tweet_eval_hate_speech[0],
        train=True,
    ),
    RunConfig(
        run_name="hate-speech-detection",
        dataset=data_tweet_eval_hate_speech[1],
        train=False,
    ),
]


cross_dataset_run_configs = [
    # RunConfig(
    #     run_name="hate-speech-detection-cross-val",
    #     dataset=data_tweet_eval_hate_speech[0],
    #     train=True,
    # ),
    RunConfig(
        run_name="hatecheck",
        dataset=data_hatecheck[1],
        train=False,
    ),
]
