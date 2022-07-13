from configs import Const
from model.pipeline import Pipeline
from model.huggingface import HuggingfaceModel

from model.sklearn import SKLearnModel
from configs.config import (
    global_preprocess_config,
    huggingface_config,
    sklearn_config,
)
from model.pipeline import Pipeline
from model.ensemble import Ensemble
from model.data import DataSource, StrConcat, VectorConcat
from model.transformations.predicitions_to_text import PredictionsToText
from model.augmenters.spellcorrector import SpellCorrectorAugmenter


def hate_speech_pipeline() -> Pipeline:
    input_data = DataSource("input")

    pipeline1 = Pipeline(
        "pipeline1",
        input_data,
        [
            SpellCorrectorAugmenter(fast=True),
            SKLearnModel("model1", sklearn_config),
            PredictionsToText(),
        ],
    )

    return pipeline1
