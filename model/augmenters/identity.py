from .base import SimpleAugmenter
import pandas as pd

class IdentityAugmenter(SimpleAugmenter):

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset
