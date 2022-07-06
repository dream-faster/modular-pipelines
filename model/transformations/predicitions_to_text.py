import pandas as pd
from .base import SimpleTransformation


class PredictionsToText(SimpleTransformation):

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset

