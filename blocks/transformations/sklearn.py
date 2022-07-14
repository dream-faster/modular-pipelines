import pandas as pd
from .base import Transformation
from configs.constants import Const
from type import BaseConfig
from sklearn.base import BaseEstimator
from utils.random import random_string
import pandas as pd
from typing import Optional, Union
import numpy as np


class SKLearnTransformation(Transformation):

    model: BaseEstimator

    def __init__(self, sklearn_transformation: BaseEstimator):
        super().__init__()
        self.model = sklearn_transformation
        self.id += "-" + sklearn_transformation.__class__.__name__

    def preload(self):
        pass

    def fit(
        self,
        dataset: Union[list, np.ndarray],
        labels: Optional[Union[list, np.ndarray]],
    ) -> None:
        self.model.fit(dataset)
        self.trained = True

    def predict(self, dataset: Union[list, np.ndarray]) -> np.ndarray:
        # TODO: this is not returning a dataframe, but a sparse vector
        return self.model.transform(dataset)

    def is_fitted(self) -> bool:
        if self.model is None:
            return False
        # source: https://stackoverflow.com/a/63839394
        attrs = [
            v for v in vars(self.model) if v.endswith("_") and not v.startswith("__")
        ]
        return len(attrs) != 0
