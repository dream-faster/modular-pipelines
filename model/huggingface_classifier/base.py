from model.base_class import BaseModel
from model.huggingface_classifier.infer import run_inference_pipeline
from model.huggingface_classifier.train import run_training_pipeline
from config import HuggingfaceConfig, huggingface_config


class HuggingfaceClassifyModel(BaseModel):
    def __init__(self, config):
        self.config = config

    def fit(self, train_dataset, val_dataset):
        return run_training_pipeline(train_dataset, val_dataset, self.config)

    def predict(self, test_dataset):
        return run_inference_pipeline(test_dataset, huggingface_config, self.config)
