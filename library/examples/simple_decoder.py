from configs import Const
from model.pipeline import Pipeline
from type import PytorchConfig
from model.huggingface import HuggingfaceModel
from model.sklearn import SKLearnModel
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
from model.adaptors import DfToNumpy


def all_transformations() -> Pipeline:

    pytorch_decoder_config = PytorchConfig(hidden_size=768, output_size=2, val_size=0.1)

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
