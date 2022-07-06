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
from model.data import DataSource, Concat
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
    "pipeline1", input_data, [SKLearnModel(config=sklearn_config)], cache=True
)

pipeline2 = Pipeline(
    "pipeline2",
    Concat([input_data, pipeline1]),
    [SKLearnModel(config=sklearn_config)],
    cache=False,
)
pipeline3 = Pipeline(
    "pipeline3",
    Concat([input_data, pipeline2]),
    [SKLearnModel(config=sklearn_config)],
    cache=False,
)


end_to_end_pipeline = Pipeline(
    "end-to-end",
    Concat([pipeline2, pipeline2, pipeline3, input_data]),
    [SKLearnModel(config=sklearn_config)],
    cache=False,
)

train_pipeline(end_to_end_pipeline, {"input": train_dataset})