from transformers import TrainingArguments
from blocks.pipeline import Pipeline
from blocks.models.huggingface import HuggingfaceModel

from blocks.models.sklearn import SKLearnModel
from library.evaluation import classification
from type import PreprocessConfig, HuggingfaceConfig, SKLearnConfig
from blocks.pipeline import Pipeline
from blocks.transformations import Lemmatizer, SpacyTokenizer
from blocks.data import DataSource
from blocks.ensemble import Ensemble
from blocks.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from blocks.transformations import SKLearnTransformation, TextStatisticTransformation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    VotingClassifier,
)
from utils.flatten import remove_none
from sklearn.preprocessing import MinMaxScaler
from blocks.adaptors import ListOfListsToNumpy

from library.evaluation import classification_metrics

preprocess_config = PreprocessConfig(
    train_size=100,
    val_size=100,
    test_size=100,
    input_col="text",
    label_col="label",
)

huggingface_config = HuggingfaceConfig(
    pretrained_model="distilbert-base-uncased",
    user_name="semy",
    repo_name="finetuning-tweeteval-hate-speech",
    save_remote=False,
    save=True,
    num_classes=2,
    val_size=0.1,
    force_fit=False,
    training_args=TrainingArguments(
        output_dir="",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=False,
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


input_data = DataSource("input")


def create_nlp_sklearn_pipeline(autocorrect: bool) -> Pipeline:
    return Pipeline(
        "nlp_sklearn_autocorrect" if autocorrect else "nlp_sklearn",
        input_data,
        remove_none(
            [
                SpellAutocorrectAugmenter(fast=True) if autocorrect else None,
                SpacyTokenizer(),
                Lemmatizer(),
                SKLearnTransformation(
                    TfidfVectorizer(
                        max_features=100000,
                        ngram_range=(1, 3),
                    )
                ),
                SKLearnModel("nlp-sklearn", sklearn_config),
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
