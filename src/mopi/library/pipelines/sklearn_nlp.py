from type import SKLearnConfig
from typing import Tuple, Union
from ..models.sklearn_voting import sklearn_config
from blocks.pipeline import Pipeline
from blocks.base import DataSource
from utils.list import remove_none
from blocks.transformations import (
    Lemmatizer,
    NoLemmatizer,
    SKLearnTransformation,
    SpacyTokenizer,
)
from blocks.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from sklearn.feature_extraction.text import TfidfVectorizer
from blocks.models.sklearn import SKLearnModel
from blocks.ensemble import Ensemble
from imblearn.over_sampling import RandomOverSampler


def create_nlp_sklearn_pipeline(
    title: str,
    input_data: DataSource,
    sklearn_config: SKLearnConfig,
    autocorrect: bool = False,
    lemmatization: bool = True,
    tfidf_ngram_range: Tuple = (1, 3),
    tfidf_min_df: Union[int, None] = 2,
    tfidf_max_features: int = 100000,
) -> Pipeline:
    return Pipeline(
        title,
        input_data,
        remove_none(
            [
                SpellAutocorrectAugmenter(fast=True) if autocorrect else None,
                SpacyTokenizer(),
                Lemmatizer(remove_stopwords=True)
                if lemmatization
                else NoLemmatizer(remove_stopwords=True),
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


def create_nlp_sklearn_ensemble_pipeline(input: DataSource) -> Pipeline:

    sklearn_no_lemma_1_3 = create_nlp_sklearn_pipeline(
        title="sklearn_no_lemma_13",
        input_data=input,
        sklearn_config=sklearn_config,
        autocorrect=True,
        lemmatization=False,
        tfidf_ngram_range=(1, 3),
        tfidf_min_df=2,
        tfidf_max_features=100000,
    )
    sklearn_lemma_1_3 = create_nlp_sklearn_pipeline(
        title="sklearn_lemma_13",
        input_data=input,
        sklearn_config=sklearn_config,
        autocorrect=False,
        lemmatization=True,
        tfidf_ngram_range=(1, 3),
        tfidf_min_df=2,
        tfidf_max_features=100000,
    )

    sklearn_lemma_1_2_small = create_nlp_sklearn_pipeline(
        title="sklearn_lemma_12",
        input_data=input,
        sklearn_config=sklearn_config,
        autocorrect=False,
        lemmatization=True,
        tfidf_ngram_range=(1, 2),
        tfidf_min_df=2,
        tfidf_max_features=10000,
    )

    sklearn_lemma_1_2_large = create_nlp_sklearn_pipeline(
        title="sklearn_lemma_12",
        input_data=input,
        sklearn_config=sklearn_config,
        autocorrect=False,
        lemmatization=True,
        tfidf_ngram_range=(1, 2),
        tfidf_min_df=2,
        tfidf_max_features=100000,
    )

    return Ensemble(
        "sklearn-ensemble",
        [
            sklearn_no_lemma_1_3,
            sklearn_lemma_1_3,
            sklearn_lemma_1_2_small,
            sklearn_lemma_1_2_large,
        ],
    )
