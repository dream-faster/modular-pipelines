from blocks.pipeline import Pipeline
from blocks.models.huggingface import HuggingfaceModel

from blocks.models.sklearn import SKLearnModel
from blocks.transformations.no_lemmatizer import NoLemmatizer
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
from typing import Tuple, Union
from library.evaluation import classification_metrics

preprocess_config = PreprocessConfig(
    train_size=100,
    val_size=100,
    test_size=100,
    input_col="text",
    label_col="label",
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
                    n_estimators=100, max_depth=7, random_state=0
                ),
            ),
        ],
        voting="soft",
    ),
    one_vs_rest=False,
    save_remote=False,
)

input_data = DataSource("input")


def create_nlp_sklearn_pipeline(
    title: str,
    autocorrect: bool,
    lemmatization: bool,
    tfidf_ngram_range: Tuple,
    tfidf_min_df: Union[int, None],
    tfidf_max_features: int,
    sklearn_config: SKLearnConfig,
) -> Pipeline:
    return Pipeline(
        title,
        input_data,
        remove_none(
            [
                SpellAutocorrectAugmenter(fast=True) if autocorrect else None,
                SpacyTokenizer(),
                Lemmatizer(remove_stopwords=False)
                if lemmatization
                else NoLemmatizer(remove_stopwords=False),
                SKLearnTransformation(
                    TfidfVectorizer(
                        max_features=tfidf_max_features,
                        min_df=1 if tfidf_min_df is None else tfidf_min_df,
                        ngram_range=tfidf_ngram_range,
                    )
                ),
                SKLearnModel(title, sklearn_config),
            ]
        ),
    )


sklearn_no_lemma_1_3 = create_nlp_sklearn_pipeline(
    title="sklearn_no_lemma_13",
    autocorrect=True,
    lemmatization=False,
    tfidf_ngram_range=(1, 3),
    tfidf_min_df=2,
    tfidf_max_features=100000,
    sklearn_config=sklearn_config,
)
sklearn_lemma_1_3 = create_nlp_sklearn_pipeline(
    title="sklearn_lemma_13",
    autocorrect=False,
    lemmatization=True,
    tfidf_ngram_range=(1, 3),
    tfidf_min_df=2,
    tfidf_max_features=100000,
    sklearn_config=sklearn_config,
)

sklearn_lemma_1_2_small = create_nlp_sklearn_pipeline(
    title="sklearn_lemma_12",
    autocorrect=False,
    lemmatization=True,
    tfidf_ngram_range=(1, 2),
    tfidf_min_df=2,
    tfidf_max_features=10000,
    sklearn_config=sklearn_config,
)

sklearn_lemma_1_2_large = create_nlp_sklearn_pipeline(
    title="sklearn_lemma_12",
    autocorrect=False,
    lemmatization=True,
    tfidf_ngram_range=(1, 2),
    tfidf_min_df=2,
    tfidf_max_features=100000,
    sklearn_config=sklearn_config,
)


sklearn_ensemble = Ensemble(
    "sklearn-ensemble",
    [
        sklearn_no_lemma_1_3,
        sklearn_lemma_1_3,
        sklearn_lemma_1_2_small,
        sklearn_lemma_1_2_large,
    ],
)
