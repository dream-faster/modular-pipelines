import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import ClassifierMixin
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from blocks.models.base import Model
from type import SKLearnConfig
from configs.constants import Const
from typing import Optional
import os
import joblib
from blocks.iomanager import safe_loading, safe_saving


class SKLearnModel(Model):

    config: SKLearnConfig

    def __init__(self, id: str, config: SKLearnConfig):
        self.id = id
        self.config = config
        self.pipeline = None

    def preload(self):
        pass

    def load(self, pipeline_id) -> None:
        self.pipeline = safe_loading(pipeline_id, self.id)

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:

        self.pipeline = ImbPipeline(
            [
                # ('feature_selection', SelectPercentile(chi2, percentile=50)),
                ("sampling", RandomOverSampler()),
                ("clf", self.config.classifier),
            ],
            verbose=True,
        )

        self.pipeline.fit(dataset, labels)

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        predictions = self.pipeline.predict(dataset)
        probabilities = [tuple(row) for row in self.pipeline.predict_proba(dataset)]

        return pd.DataFrame(
            {Const.preds_col: predictions, Const.probs_col: probabilities}
        )

    def is_fitted(self) -> bool:
        return hasattr(self, "pipeline") and self.pipeline is not None

    def save(self, pipeline_id: str) -> None:
        safe_saving(self.pipeline, pipeline_id, self.id)


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
