from configs import Const
from model.pipeline import Pipeline
from model.huggingface import HuggingfaceModel

from model.sklearn import SKLearnModel
from data.dataloader import load_data
from configs.config import (
    global_preprocess_config,
    huggingface_config,
    sklearn_config,
    GlobalPreprocessConfig,
)
from model.pipeline import Pipeline
from model.ensemble import Ensemble
from model.data import DataSource, StrConcat, VectorConcat
from model.transformations.predicitions_to_text import PredictionsToText
import pandas as pd
from runner.run import train_pipeline

train_dataset, test_dataset = load_data("data/original", global_preprocess_config)


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
    PytorchModel("decoder"),
)


end_to_end_pipeline = Ensemble(
    "end-to-end",
    VectorConcat([nlp_pipeline, image_pipeline, multimodal_pipeline]),
    [
        PytorchModel("last_model_1"),
        PytorchModel("last_model_2"),
        PytorchModel("last_model_3"),
    ],
)


####

input_data = DataSource("input")

pipeline1 = Pipeline(
    "pipeline1",
    input_data,
    [SKLearnModel("model1", sklearn_config), PredictionsToText()],
)

pipeline2 = Pipeline(
    "pipeline2",
    StrConcat([input_data, pipeline1]),
    [SKLearnModel("model2", sklearn_config), PredictionsToText()],
)
pipeline3 = Pipeline(
    "pipeline3",
    StrConcat([input_data, pipeline2]),
    [SKLearnModel("model3", sklearn_config), PredictionsToText()],
)


end_to_end_pipeline = Pipeline(
    "end-to-end",
    StrConcat([input_data, pipeline1, pipeline2, pipeline3]),
    [SKLearnModel("decoder", sklearn_config)],
)

train_pipeline(
    end_to_end_pipeline, {"input": train_dataset}, train_dataset[Const.label_col]
)
