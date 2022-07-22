import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import ClassifierMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from blocks.models.base import Model
from type import Evaluators, PredsWithProbs, SKLearnConfig, DataType
from configs.constants import Const
from typing import Optional, List
from sklearn.base import clone, BaseEstimator


class SKLearnModel(Model):

    config: SKLearnConfig
    model: Optional[BaseEstimator] = None

    inputTypes = [DataType.Series, DataType.List, DataType.NpArray]
    outputType = DataType.PredictionsWithProbs

    def __init__(
        self, id: str, config: SKLearnConfig, evaluators: Optional[Evaluators] = None
    ):
        self.id = id
        self.config = config
        self.evaluators = evaluators

    def fit(self, dataset: pd.Series, labels: Optional[pd.Series]) -> None:
        self.model = clone(self.config.classifier)
        self.model.fit(dataset, labels)

    def predict(self, dataset: pd.Series) -> List[PredsWithProbs]:
        predictions = self.model.predict(dataset)
        probabilities = self.model.predict_proba(dataset)

        return list(zip(predictions, probabilities))

    def is_fitted(self) -> bool:
        return self.model is not None


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
