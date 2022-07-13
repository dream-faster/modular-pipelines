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
from model.augmenters.statistics import StatisticAugmenter


def simple_decoder() -> Pipeline:

    nlp_input = DataSource("input")

    stacked_pipeline = Pipeline(
        "stacked",
        nlp_input,
        models=[
            StatisticAugmenter("statistic-augmenter", config=None),
            # HuggingfaceModel("transformer1", huggingface_config),
            PytorchModel(id="pytorch-decoder", config=pytorch_decoder_config),
        ],
    )

    return stacked_pipeline
