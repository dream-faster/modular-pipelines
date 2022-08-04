from copy import deepcopy


from blocks.adaptors import ListOfListsToNumpy
from blocks.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from blocks.concat import DataSource, VectorConcat

from blocks.ensemble import Ensemble
from blocks.models.huggingface import HuggingfaceModel
from blocks.pipeline import Pipeline

from blocks.transformations import (
    Lemmatizer,
    SKLearnTransformation,
    SpacyTokenizer,
    TextStatisticTransformation,
)
from configs.constants import Const
from data.transformation import transform_dataset
from datasets.load import load_dataset
from library.evaluation import classification_metrics  # , calibration_metrics
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from transformers import TrainingArguments
from type import (
    DatasetSplit,
    HuggingfaceConfig,
    LoadOrigin,
    PreprocessConfig,
    Experiment,
    SKLearnConfig,
    HFTaskTypes,
)
from blocks.models.sklearn import SKLearnModel
from blocks.adaptors.classification_output import ClassificationOutputAdaptor

from utils.flatten import remove_none
from data.dataloader import DataLoader

""" Models """
preprocess_config = PreprocessConfig(
    train_size=-1,
    val_size=-1,
    test_size=-1,
    input_col="text",
    label_col="label",
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
                    n_estimators=100, max_depth=7, random_state=0
                ),
            ),
        ],
        voting="soft",
    ),
    one_vs_rest=False,
    save_remote=False,
)

""" Data """

input_data = DataSource("input")

dataloader = DataLoader("tweet_eval", preprocess_config, [transform_dataset], "hate")


""" Pipelines"""

sklearn_first = Pipeline(
    "nlp_hf_distilbert",
    input_data,
    remove_none(
        [
            SpacyTokenizer(),
            Lemmatizer(remove_stopwords=False),
            SKLearnTransformation(
                TfidfVectorizer(
                    max_features=100000,
                    ngram_range=(1, 3),
                )
            ),
            SKLearnModel("sklearn-model", sklearn_config),
            ClassificationOutputAdaptor(select=0),
        ]
    ),
)

sklearn_second = Pipeline(
    "nlp_hf_distilroberta-base",
    input_data,
    remove_none(
        [
            SpacyTokenizer(),
            Lemmatizer(remove_stopwords=False),
            SKLearnTransformation(
                TfidfVectorizer(
                    max_features=100000,
                    ngram_range=(1, 3),
                )
            ),
            SKLearnModel("sklearn-model", sklearn_config),
            ClassificationOutputAdaptor(select=0),
        ]
    ),
)


full_pipeline = Pipeline(
    "nlp_hf_meta-model-pipeline",
    VectorConcat("concat-source", [sklearn_first, sklearn_second]),
    remove_none(
        [
            SKLearnModel("sklearn-meta-model", sklearn_config),
        ]
    ),
)

metrics = classification_metrics  # + calibration_metrics

""" Experiments """
multi_sklearn_run_experiments = [
    Experiment(
        project_name="hate-speech-detection-hf",
        run_name="hf-meta-model",
        dataloader=dataloader,
        dataset_category=DatasetSplit.train,
        pipeline=full_pipeline,
        preprocessing_config=preprocess_config,
        metrics=metrics,
        train=True,
    ),
    Experiment(
        project_name="hate-speech-detection-hf",
        run_name="hf-meta-model",
        dataloader=dataloader,
        dataset_category=DatasetSplit.test,
        pipeline=full_pipeline,
        preprocessing_config=preprocess_config,
        metrics=metrics,
        train=False,
    ),
]
