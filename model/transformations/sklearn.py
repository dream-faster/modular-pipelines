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

    transformation: BaseEstimator

    def __init__(self, sklearn_transformation: BaseEstimator):
        super().__init__()
        self.transformation = sklearn_transformation

    def preload(self):
        pass

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        self.transformation.fit(dataset)

    def predict(self, dataset: Union[list, np.ndarray]) -> np.ndarray:
        # TODO: this is not returning a dataframe, but a sparse vector
        return self.transformation.transform(dataset)

    def is_fitted(self) -> bool:
        # source: https://stackoverflow.com/a/63839394
        attrs = [
            v
            for v in vars(self.transformation)
            if v.endswith("_") and not v.startswith("__")
        ]
        return len(attrs) != 0
