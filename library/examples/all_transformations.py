from configs import Const
from blocks.pipeline import Pipeline
from type import PytorchConfig
from blocks.models.huggingface import HuggingfaceModel
from blocks.models.sklearn import SKLearnModel
from blocks.pipeline import Pipeline
from blocks.ensemble import Ensemble
from blocks.data import DataSource, StrConcat, VectorConcat
from blocks.transformations.predicitions_to_text import PredictionsToText
from blocks.models.pytorch.base import PytorchModel
from blocks.augmenters import StatisticAugmenter, SynonymAugmenter

from blocks.transformations import PredictionsToText, SpacyTokenizer
from blocks.transformations.sklearn import SKLearnTransformation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from blocks.adaptors import DfToNumpy


def all_transformations() -> Pipeline:
    nlp_input = DataSource("input")

    stacked_pipeline = Pipeline(
        "stacked",
        nlp_input,
        models=[
            SpacyTokenizer(),
            StatisticAugmenter(),
            DfToNumpy(),
            SKLearnTransformation(MinMaxScaler(feature_range=(0, 1), clip=True)),
        ],
    )

    return stacked_pipeline
