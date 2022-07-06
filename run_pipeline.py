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

global_data = DataSource("global")

model1 = Pipeline("model1", global_data, [SKLearnModel(config=sklearn_config)])
model1_alt = Pipeline("model1_alt", model1, [XYTRansformation())])

pipeline1 = SKLearnModel(config=sklearn_config)
pipeline2 = Pipeline(
    "model2", Concat([global_data, model1_alt]), [pipeline1]
)
model3 = Pipeline(
    "model3", Concat([global_data, pipeline2]), [SKLearnModel(config=sklearn_config)]
)


end_to_end_pipeline = Pipeline(
    "end-to-end",
    Concat([model1, model2, model3, global_data]),
    [SKLearnModel(config=sklearn_config)],
)

def run_pipeline(pipeline: Pipeline, data: Dict[pd.DataFrame]) -> pd.DataFrame:
    pipeline.run(data)
    return data