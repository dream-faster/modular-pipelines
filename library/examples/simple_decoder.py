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


def simple_decoder() -> Pipeline:

    pytorch_decoder_config = PytorchConfig(hidden_size=768, output_size=2, val_size=0.1)

    nlp_input = DataSource("input")

    stacked_pipeline = Pipeline(
        "stacked",
        nlp_input,
        models=[
            SpacyTokenizer(),
            StatisticAugmenter(),
            # HuggingfaceModel("transformer1", huggingface_config),
            PytorchModel(id="pytorch-decoder", config=pytorch_decoder_config),
        ],
    )

    return stacked_pipeline