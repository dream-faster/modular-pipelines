from .base import BaseModel
import pandas as pd
from type import Label, Probabilities, BaseConfig
from typing import List, Tuple
import numpy as np
from training.train import train_model


class Ensemble(BaseModel):
    def __init__(self, models: List[BaseModel]):
        self.models = models
        self.config = BaseConfig(force_fit=False)

    def preload(self):
        for model in self.models:
            model.preload()

    def fit(self, train_dataset: pd.DataFrame) -> None:
        for model in self.models:
            train_model(model, train_dataset)

    def predict(self, test_dataset: pd.DataFrame) -> List[Tuple[Label, Probabilities]]:
        probabilities = np.average(
            np.array(
                [
                    [item[1] for item in model.predict(test_dataset)]
                    for model in self.models
                ]
            ),
            axis=0,
        )
        predictions = [np.argmax(probs) for probs in probabilities]
        return list(zip(predictions, probabilities))

    def is_fitted(self) -> bool:
        return all([model.is_fitted() for model in self.models])
