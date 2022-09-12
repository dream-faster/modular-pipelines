import random
from typing import List, Optional

import pandas as pd

from blocks.models.base import Model
from type import DataType, PredsWithProbs


class AllZerosModel(Model):

    inputTypes = [DataType.Series, DataType.List, DataType.NpArray]
    outputType = DataType.PredictionsWithProbs

    def fit(self, dataset: pd.Series, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.Series) -> List[PredsWithProbs]:
        predictions = [0 for _ in dataset]
        probabilities = [[1.0, 0.0] for _ in dataset]
        return list(zip(predictions, probabilities))

    def is_fitted(self) -> bool:
        return True

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass


class AllOnesModel(Model):

    inputTypes = [DataType.Series, DataType.List, DataType.NpArray]
    outputType = DataType.PredictionsWithProbs

    def fit(self, dataset: pd.Series, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.Series) -> List[PredsWithProbs]:
        predictions = [1 for _ in dataset]
        probabilities = [[0.0, 1.0] for _ in dataset]
        return list(zip(predictions, probabilities))

    def is_fitted(self) -> bool:
        return True

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass


class RandomModel(Model):

    inputTypes = [DataType.Series, DataType.List, DataType.NpArray]
    outputType = DataType.PredictionsWithProbs

    def fit(self, dataset: pd.Series, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.Series) -> List[PredsWithProbs]:
        predictions = [random.randint(0, 1) for _ in dataset]
        probabilities = [
            [0.0 if item == 1 else 1.0, 1.0 if item == 1 else 0.0]
            for item in predictions
        ]
        return list(zip(predictions, probabilities))

    def is_fitted(self) -> bool:
        return True

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass
