from configs.constants import Const
from .base import Block, Model
from .data import DataSource
import pandas as pd
from typing import List
from runner.train import train_predict, predict
from runner.store import Store
import numpy as np


class Ensemble(Block):

    id: str
    datasource: DataSource
    models: List[Model]

    def __init__(self, id: str, datasource: DataSource, models: List[Model]):
        self.id = id
        self.models = models
        self.datasource = datasource

    def preload(self):
        self.datasource.preload()
        for model in self.models:
            model.preload()

    def fit(self, store: Store) -> None:
        input = self.datasource.deplate(store)
        for model in self.models:
            output = train_predict(model, input, store)
            store.set_data(model.id, output)

    def predict(self, store: Store) -> pd.DataFrame:
        input = self.datasource.deplate(store)
        outputs: List[pd.DataFrame] = []
        for model in self.models:
            output = predict(model, input)
            outputs.append(output)

        averaged = average_output(outputs)
        store.set_data(self.id, averaged)
        return averaged

    def is_fitted(self) -> bool:
        return all([model.is_fitted() for model in self.models])


def average_output(outputs: List[pd.DataFrame]) -> pd.DataFrame:
    probabilities = np.average(
        np.array([item[1] for item in outputs]),
        axis=0,
    )
    predictions = [np.argmax(probs) for probs in probabilities]
    return pd.DataFrame({Const.preds_col: predictions, Const.probs_col: probabilities})
