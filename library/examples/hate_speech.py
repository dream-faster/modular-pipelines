from transformers import TrainingArguments
from blocks.pipeline import Pipeline
from blocks.models.huggingface import HuggingfaceModel

from blocks.models.sklearn import SKLearnModel
from library.evaluation.classification import classification_metrics
from type import PreprocessConfig, HuggingfaceConfig, SKLearnConfig
from blocks.pipeline import Pipeline
from blocks.adaptors import SeriesToList
from blocks.transformations import Lemmatizer, SpacyTokenizer
from blocks.data import DataSource, StrConcat, VectorConcat
from blocks.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from blocks.transformations.sklearn import SKLearnTransformation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    VotingClassifier,
)

preprocess_config = PreprocessConfig(
    train_size=-1,
    val_size=-1,
    test_size=-1,
    input_col="text",
    label_col="label",
)

huggingface_config = HuggingfaceConfig(
    pretrained_model="distilbert-base-uncased",
    user_name="semy",
    repo_name="finetuning-tweeteval-hate-speech",
    save_remote=True,
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

nb = MultinomialNB()
lg = LogisticRegression()
gb = GradientBoostingClassifier(n_estimators=100, max_depth=20, random_state=0)

sklearn_config = SKLearnConfig(
    force_fit=False,
    save=True,
    classifier=VotingClassifier(
        estimators=[("nb", nb), ("lg", lg), ("gb", gb)], voting="soft"
    ),
    one_vs_rest=False,
    save_remote=False,
)


input_data = DataSource("input")


nlp_sklearn = Pipeline(
    "nlp_sklearn",
    input_data,
    [
        SeriesToList(),
        SpacyTokenizer(),
        Lemmatizer(),
        SKLearnTransformation(
            TfidfVectorizer(
                max_features=100000,
                ngram_range=(1, 3),
            )
        ),
        SKLearnModel("model1", sklearn_config, evaluators=classification_metrics),
    ],
)

nlp_sklearn_autocorrect = Pipeline(
    "nlp_sklearn_autocorrect",
    input_data,
    [
        SeriesToList(),
        SpellAutocorrectAugmenter(fast=True),
        SpacyTokenizer(),
        Lemmatizer(),
        SKLearnTransformation(
            TfidfVectorizer(
                max_features=100000,
                ngram_range=(1, 3),
            )
        ),
        SKLearnModel("model1", sklearn_config),
    ],
)


nlp_huggingface = Pipeline(
    "nlp_huggingface",
    input_data,
    [
        HuggingfaceModel("hf-model-full", huggingface_config),
    ],
)

nlp_huggingface_autocorrect = Pipeline(
    "nlp_huggingface_autocorrect",
    input_data,
    [
        SpellAutocorrectAugmenter(fast=True),
        HuggingfaceModel("hf-model", huggingface_config),
    ],
)


def hate_speech_detection_pipeline() -> Pipeline:
    return nlp_huggingface
