from .base import Augmenter
import pandas as pd


class IdentityAugmenter(Augmenter):
    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset
