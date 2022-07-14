from configs import Const
from model.pipeline import Pipeline

from model.huggingface import HuggingfaceModel
from model.sklearn import SKLearnModel

from configs.config import (
    global_preprocess_config,
    huggingface_config,
    sklearn_config,
    pytorch_decoder_config,
)
from model.pipeline import Pipeline
from model.ensemble import Ensemble
from model.data import DataSource, StrConcat, VectorConcat
from model.transformations.predicitions_to_text import PredictionsToText
from model.pytorch.base import PytorchModel
from model.augmenters import StatisticAugmenter, SynonymAugmenter

from model.transformations import PredictionsToText, SpacyTokenizer
from model.transformations.sklearn import SKLearnTransformation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def all_transformations() -> Pipeline:

    nlp_input = DataSource("input")

    stacked_pipeline = Pipeline(
        "stacked",
        nlp_input,
        models=[
            SpacyTokenizer(),
            StatisticAugmenter(),
            SKLearnTransformation(MinMaxScaler(feature_range=(0, 1), clip=True)),
            SKLearnTransformation(
                TfidfVectorizer(
                    max_features=100000,
                    ngram_range=(1, 3),
                )
            ),
        ],
    )

    return stacked_pipeline
