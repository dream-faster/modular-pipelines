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
from model.pytorch.decoder import Decoder


def text_image_pipeline() -> Pipeline:
    nlp_input = DataSource("text")
    image_input = DataSource("image")

    nlp_pipeline = Pipeline(
        "nlp", nlp_input, HuggingfaceModel("transformer1", huggingface_config)
    )
    image_pipeline = Pipeline(
        "image", image_input, HuggingfaceModel("cnn1", huggingface_config)
    )

    multimodal_pipeline = Pipeline(
        "multimodal",
        VectorConcat(
            [
                Pipeline(
                    "multimodal-nlp",
                    nlp_input,
                    HuggingfaceModel("transformer1", huggingface_config),
                ),
                Pipeline(
                    "multimodal-image",
                    image_input,
                    HuggingfaceModel("cnn2", huggingface_config),
                ),
            ]
        ),
        Decoder("pytroch-decoder", pytorch_decoder_config),
    )

    # end_to_end_pipeline = Ensemble(
    #     "end-to-end",
    #     VectorConcat([nlp_pipeline, image_pipeline, multimodal_pipeline]),
    #     [
    #         PytorchModel("last_model_1"),
    #         PytorchModel("last_model_2"),
    #         PytorchModel("last_model_3"),
    #     ],
    # )

    return multimodal_pipeline
