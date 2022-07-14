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


class SKLearnModel(Model):

    config: SKLearnConfig

    def __init__(self, id: str, config: SKLearnConfig):
        self.id = id
        self.config = config
        self.pipeline = None

    def preload(self):
        pass

    def load(self, pipeline_id) -> None:
        path = f"output/pipelines/{pipeline_id}/{self.id}.pkl"
        if os.path.exists(path):
            print(f"| Loading model {pipeline_id}/{self.id}")
            with open(path, "rb") as f:
                self.pipeline = joblib.load(f)

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
        path = f"output/pipelines/{pipeline_id}"
        if os.path.exists(path) is False:
            os.makedirs(path)

        with open(path + f"/{self.id}.pkl", "wb") as f:
            joblib.dump(self.pipeline, f, compress=9)


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
