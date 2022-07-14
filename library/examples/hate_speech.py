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
from model.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from model.transformations.sklearn import SKLearnTransformation
from sklearn.feature_extraction.text import TfidfVectorizer


def hate_speech_pipeline() -> Pipeline:
    input_data = DataSource("input")

    pipeline1 = Pipeline(
        "pipeline1",
        input_data,
        [
            # SpellAutocorrectAugmenter(fast=True),
            SKLearnTransformation(
                TfidfVectorizer(
                    max_features=100000,
                    ngram_range=(1, 3),
                )
            ),
            SKLearnModel("model1", sklearn_config),
            PredictionsToText(),
        ],
    )

    return pipeline1
