from configs.constants import Const
from .base import Block, Element
from .data import DataSource
import pandas as pd
from typing import List
from runner.train import train_predict, predict
from runner.store import Store
import numpy as np
from type import PredsWithProbs


class Ensemble(Block):

    id: str
    datasource: DataSource
    models: List[Block]

    def __init__(self, id: str, datasource: DataSource, models: List[Block]):
        self.id = id
        self.models = models
        self.datasource = datasource

    def load_remote(self):
        self.datasource.load_remote()
        for model in self.models:
            model.load_remote()

    def fit(self, store: Store) -> None:
        input = self.datasource.deplate(store)
        for model in self.models:
            output = train_predict(model, input, store)
            store.set_data(model.id, output)

    def predict(self, store: Store) -> List[PredsWithProbs]:
        input = self.datasource.deplate(store)
        outputs: List[pd.Series] = []
        for model in self.models:
            output = predict(model, input)
            outputs.append(output)

        averaged = average_output(outputs)
        store.set_data(self.id, averaged)
        return averaged

    def is_fitted(self) -> bool:
        return all([model.is_fitted() for model in self.models])

    def children(self) -> List[Element]:
        return [self.datasource] + [self] + [self.models]


def average_output(outputs: List[pd.Series]) -> List[PredsWithProbs]:
    probabilities = np.average(
        np.array([item[1] for item in outputs]),
        axis=0,
    )
    predictions = [np.argmax(probs) for probs in probabilities]
    return zip(predictions, probabilities)
