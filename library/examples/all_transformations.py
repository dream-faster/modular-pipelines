from configs import Const
from blocks.pipeline import Pipeline
from type import PytorchConfig
from blocks.models.huggingface import HuggingfaceModel
from blocks.models.sklearn import SKLearnModel
from blocks.pipeline import Pipeline
from blocks.ensemble import Ensemble
from blocks.data import DataSource, StrConcat, VectorConcat
from blocks.models.pytorch.base import PytorchModel
from blocks.augmenters import StatisticAugmenter, SynonymAugmenter

from blocks.transformations import SpacyTokenizer
from blocks.transformations.sklearn import SKLearnTransformation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from blocks.adaptors import ListOfListsToNumpy


def all_transformations() -> Pipeline:
    nlp_input = DataSource("input")

    stacked_pipeline = Pipeline(
        "stacked",
        nlp_input,
        models=[
            SpacyTokenizer(),
            StatisticAugmenter(),
            ListOfListsToNumpy(),
            SKLearnTransformation(MinMaxScaler(feature_range=(0, 1), clip=True)),
        ],
    )

    return stacked_pipeline
