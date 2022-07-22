from .base import Block, Element
from .pipeline import Pipeline
import pandas as pd
from typing import List
from runner.store import Store
import numpy as np
from type import PredsWithProbs


class Ensemble(Pipeline):
    def __init__(self, id: str, pipelines: List[Pipeline]):
        self.id = id
        self.pipelines = pipelines

    def load(self, plugins: List["Plugin"]) -> None:
        for pipeline in self.pipelines:
            pipeline.load(plugins)

    def load_remote(self):
        for pipeline in self.pipelines:
            pipeline.load_remote()

    def save_remote(self) -> None:
        for pipeline in self.pipelines:
            pipeline.save_remote()

    def fit(self, store: Store, plugins: List["Plugin"]) -> None:
        for pipeline in self.pipelines:
            pipeline.fit(store, plugins)

    def predict(self, store: Store, plugins: List["Plugin"]) -> pd.Series:
        outputs: List[List[PredsWithProbs]] = []
        for pipeline in self.pipelines:
            output = pipeline.predict(store, plugins)
            outputs.append(output)

        averaged = average_output(outputs)
        store.set_data(self.id, averaged)
        return averaged

    def is_fitted(self) -> bool:
        return all([pipeline.is_fitted() for pipeline in self.pipelines])

    def children(self) -> List[Element]:
        return [self] + [pipeline.children() for pipeline in self.pipelines]


def average_output(outputs: List[List[PredsWithProbs]]) -> List[PredsWithProbs]:
    probabilities = np.average(
        np.array([[item[1] for item in tuples] for tuples in list(zip(*outputs))]),
        axis=1,
    )
    predictions = [np.argmax(probs) for probs in probabilities]
    return list(zip(predictions, probabilities))
