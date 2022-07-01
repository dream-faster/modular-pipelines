from model.base_class import BaseModel
from model.huggingface_classifier.infer import run_inference_pipeline
from model.huggingface_classifier.train import run_training_pipeline
from config import HuggingfaceConfig, huggingface_config


class HuggingfaceClassifyModel(BaseModel):
    def __init__(self, data, config):
        self.train_data, self.val_data, self.test_data = data[0], data[1], data[2]
        self.config = config

    def fit(self):
        return run_training_pipeline(self.train_data, self.val_data, self.config)

    def predict(self):
        return run_inference_pipeline(self.test_data, huggingface_config, self.config)
