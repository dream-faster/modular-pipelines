from typing import List, Union, Optional

import numpy as np
import pandas as pd

from mopi.runner.store import Store
from mopi.type import PredsWithProbs, Hierarchy, SourceTypes

from .base import Block, Element, DataSource
from .pipeline import Pipeline


class Ensemble(Pipeline):
    def __init__(
        self,
        id: str,
        datasource: Union[DataSource, "Pipeline", "Concat"],
        pipelines: List[Pipeline],
        datasource_predict: Optional[Union[DataSource, "Pipeline", "Concat"]] = None,
    ):
        self.id = id
        self.pipelines = pipelines
        self.datasource = datasource
        if datasource_predict is None:
            self.datasource_predict = datasource
        else:
            self.datasource_predict = datasource_predict

    def load(self, plugins: List["Plugin"]) -> None:
        for pipeline in self.pipelines:
            pipeline.load(plugins)

    def save_remote(self) -> None:
        for pipeline in self.pipelines:
            pipeline.save_remote()

    def fit(self, store: Store, plugins: List["Plugin"]) -> None:
        for pipeline in self.pipelines:
            pipeline.fit(store, plugins)

    def predict(self, store: Store, plugins: List["Plugin"]) -> List[PredsWithProbs]:
        outputs: List[List[PredsWithProbs]] = []
        for pipeline in self.pipelines:
            output = pipeline.predict(store, plugins)
            outputs.append(output)

        averaged = average_output(outputs)
        store.set_data(self.id, averaged)
        return averaged

    def is_fitted(self) -> bool:
        return all([pipeline.is_fitted() for pipeline in self.pipelines])

    def children(self, source_type: SourceTypes) -> List[Element]:
        return [self] + [pipeline.children(source_type) for pipeline in self.pipelines]

    def get_hierarchy(self, source_hierarchy: Hierarchy) -> Hierarchy:
        return Hierarchy(
            name=self.id,
            obj=self,
            children=[child.get_hierarchy(source_hierarchy) for child in self.pipelines]
            if hasattr(self, "pipelines")
            else [],
        )


def average_output(outputs: List[List[PredsWithProbs]]) -> List[PredsWithProbs]:
    probabilities = np.average(
        np.array([[item[1] for item in tuples] for tuples in list(zip(*outputs))]),
        axis=1,
    )
    predictions = [np.argmax(probs) for probs in probabilities]
    return list(zip(predictions, probabilities))
