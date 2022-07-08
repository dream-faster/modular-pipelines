from model.pipeline import Pipeline
from model.huggingface import HuggingfaceModel

from configs.config import huggingface_config
from model.pipeline import Pipeline
from model.data import DataSource


nlp_pipeline = Pipeline(
    "nlp", DataSource("text"), HuggingfaceModel("transformer1", huggingface_config)
)
