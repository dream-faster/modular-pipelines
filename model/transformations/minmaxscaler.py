import pandas as pd
from .base import Transformation
from configs.constants import Const
from type import BaseConfig
from sklearn.base import BaseEstimator
from utils.random import random_string
import pandas as pd
from typing import Optional
from sklearn import SKLearnTransformation
from sklearn.preprocessing import MinMaxScaler

class MinMaxScaler(Transformation):
    def __init__(self, sklearn_transformation: BaseEstimator):
        super().__init__()
        self.config = BaseConfig(force_fit=False)
        self.transformation = sklearn_transformation

    def preload(self):
        pass

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        
        dataset[Const.input_col] = dataset[Const.input_col].apply(lambda x: )
        
        self.transformation.fit(dataset)

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # TODO: this is not returning a dataframe, but a sparse vector
        return self.transformation.transform(dataset[Const.input_col])

    def is_fitted(self) -> bool:
        # source: https://stackoverflow.com/a/63839394
        attrs = [
            v
            for v in vars(self.transformation)
            if v.endswith("_") and not v.startswith("__")
        ]
        return len(attrs) != 0


class SKLearnTransformation(Transformation):

    transformation: BaseEstimator

    def __init__(self, sklearn_transformation: BaseEstimator):
        self.config = BaseConfig(force_fit=False)
        self.id = sklearn_transformation.__class__.__name__ + "-" + random_string(5)
        self.transformation = sklearn_transformation

    def preload(self):
        pass

    def fit(self, dataset: pd.DataFrame, labels: Optional[pd.Series]) -> None:
        self.transformation.fit(dataset)

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # TODO: this is not returning a dataframe, but a sparse vector
        return self.transformation.transform(dataset[Const.input_col])

    def is_fitted(self) -> bool:
        # source: https://stackoverflow.com/a/63839394
        attrs = [
            v
            for v in vars(self.transformation)
            if v.endswith("_") and not v.startswith("__")
        ]
        return len(attrs) != 0
