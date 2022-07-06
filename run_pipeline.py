from typing import Dict, List

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
from model.base import Block
from model.pipeline import Pipeline
from model.augmenters.identity import IdentityAugmenter
from model.data import DataSource, StrConcat
from model.transformations.predicitions_to_text import PredictionsToText
from runner.train import train_model
import pandas as pd
from runner.run import train_pipeline

train_dataset, test_dataset = load_data("data/original", global_preprocess_config)

# augmenter = IdentityAugmenter()
# graph_augmenter =
# model = SKLearnModel(config=sklearn_config)
# graph_model =

# nlp_input = DataBlock("text")
# graph_input = DataBlock("graph")
# tabular_input = DataBlock("tabular")
# nlp_pipeline = Pipeline("nlp", nlp_input, [augmenter, model])

# graph_pipeline = Pipeline("graph", graph_input, [graph_augmenter, graph_model])

# decoder = PytorchModel()


# end_to_end_pipeline = Pipeline(
#     "end-to-end", Concat([nlp_pipeline, graph_pipeline, tabular_input]), [decoder]
# )

# pipeline.run(dict: Dict[pd.DataFrame])


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
    StrConcat([pipeline1, pipeline2, pipeline3, input_data]),
    [SKLearnModel("decoder", sklearn_config)],
)

train_pipeline(end_to_end_pipeline, {"input": train_dataset})
