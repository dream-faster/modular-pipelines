from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.multiclass import OneVsRestClassifier

from blocks.models.base import Model
from type import DataType, Evaluators, PredsWithProbs, SKLearnConfig
from ..io import PickleIO


class SKLearnModel(PickleIO, Model):

    config: SKLearnConfig
    model: Optional[BaseEstimator] = None

    inputTypes = [DataType.Series, DataType.List, DataType.NpArray]
    outputType = DataType.PredictionsWithProbs

    trained = False

    def __init__(
        self, id: str, config: SKLearnConfig, evaluators: Optional[Evaluators] = None
    ):
        self.id = id
        self.config = config
        self.evaluators = evaluators

    def fit(self, dataset: pd.Series, labels: Optional[pd.Series]) -> None:
        self.model = clone(self.config.classifier)
        self.model.fit(dataset, labels)
        self.trained = True

    def predict(self, dataset: pd.Series) -> List[PredsWithProbs]:
        predictions = self.model.predict(dataset)
        probabilities = self.model.predict_proba(dataset)

        return list(zip(predictions, probabilities))

    def is_fitted(self) -> bool:
        return self.trained


def create_classifier(
    classifier: ClassifierMixin, one_vs_rest: bool
) -> ClassifierMixin:
    if one_vs_rest:
        return classifier
    else:
        return OneVsRestClassifier(
            ("clf", classifier),
            n_jobs=4,
        )
