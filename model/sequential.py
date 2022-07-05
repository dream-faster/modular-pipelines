from configs.constants import Const
from .base import BaseModel
import pandas as pd
from type import BaseConfig
from typing import List, Any
import numpy as np
from training.train import train_model


class Sequential(BaseModel):
    def __init__(self, models: List[BaseModel]):
        self.models = models
        self.config = BaseConfig(force_fit=False)

    def preload(self):
        for model in self.models:
            model.preload()

    def fit(self, train_dataset: pd.DataFrame) -> None:
        last_output = train_dataset
        for model in self.models:
            train_model(model, last_output)
            last_output = pd.DataFrame(
                {
                    Const.input_col: model.predict(last_output[Const.input_col]),
                    Const.label_col: train_dataset[Const.label_col],
                }
            )

    def predict(self, test_dataset: List[Any]) -> List[Any]:
        last_output = test_dataset
        for model in self.models:
            last_output = model.predict(test_dataset)
        return last_output

    def is_fitted(self) -> bool:
        return all([model.is_fitted() for model in self.models])
