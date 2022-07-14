from model.pipeline import Pipeline
from model.huggingface import HuggingfaceModel

from model.sklearn import SKLearnModel
from type import PreprocessConfig, HuggingfaceConfig, SKLearnConfig
from model.pipeline import Pipeline
from model.transformations import Lemmatizer, SpacyTokenizer
from model.data import DataSource, StrConcat, VectorConcat
from model.transformations.predicitions_to_text import PredictionsToText
from model.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from model.transformations.sklearn import SKLearnTransformation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    VotingClassifier,
)

preprocess_config = PreprocessConfig(
    train_size=10,
    val_size=10,
    test_size=5,
    input_col="text",
    label_col="label",
)

huggingface_config = HuggingfaceConfig(
    pretrained_model="distilbert-base-uncased",
    epochs=2,
    user_name="itchingpixels",
    repo_name="finetuning-tweeteval-hate-speech",
    push_to_hub=True,
    num_classes=2,
    val_size=0.1,
    force_fit=False,
)

nb = MultinomialNB()
lg = LogisticRegression()
gb = GradientBoostingClassifier(n_estimators=100, max_depth=20, random_state=0)

sklearn_config = SKLearnConfig(
    force_fit=False,
    classifier=VotingClassifier(
        estimators=[("nb", nb), ("lg", lg), ("gb", gb)], voting="soft"
    ),
    one_vs_rest=False,
)


input_data = DataSource("input")

nlp_sklearn = Pipeline(
    "nlp_sklearn",
    input_data,
    [
        SpacyTokenizer(),
        Lemmatizer(),
        SKLearnTransformation(
            TfidfVectorizer(
                max_features=100000,
                ngram_range=(1, 3),
            )
        ),
        SKLearnModel("model1", sklearn_config),
        PredictionsToText(),
    ],
)
nlp_sklearn_autocorrect = Pipeline(
    "nlp_sklearn_autocorrect",
    input_data,
    [
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
        PredictionsToText(),
    ],
)


nlp_huggingface = Pipeline(
    "nlp_huggingface",
    input_data,
    [
        HuggingfaceModel("hf-model", huggingface_config),
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
    return nlp_sklearn_autocorrect