from typing import List, Optional

import pandas as pd


from blocks.models.base import Model
from type import DataType, Evaluators, PredsWithProbs, SKLearnConfig
from ...io import PickleIO

from perspective import PerspectiveAPI
from utils.env_interface import get_env


class PerspectiveModel(PickleIO, Model):

    inputTypes = [DataType.Series, DataType.List, DataType.NpArray]
    outputType = DataType.PredictionsWithProbs

    trained = False

    def __init__(self, id: str, evaluators: Optional[Evaluators] = None):
        self.id = id
        self.evaluators = evaluators
        self.trained = True

        self.model = PerspectiveAPI(get_env("PERSPECTIVE_TOKEN"))

    def fit(self, dataset: pd.Series, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.Series) -> List[PredsWithProbs]:

        # results = [
        #     (
        #         0 if result["TOXICITY"] < 0.5 else 1,
        #         [1 - result["TOXICITY"], result["TOXICITY"]],
        #     )
        #     for text in dataset
        #     for result in [self.model.score(text)]
        # ]

        results = [
            (
                0 if result["TOXICITY"] < 0.5 else 1,
                [1 - result["TOXICITY"], result["TOXICITY"]],
            )
            for result in [self.model.score(text) for text in dataset]
        ]

        return results

    def is_fitted(self) -> bool:
        return self.trained
