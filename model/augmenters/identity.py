from .base import BaseAugmenter
import pandas as pd


class IdentityAugmenter(BaseAugmenter):
    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset
