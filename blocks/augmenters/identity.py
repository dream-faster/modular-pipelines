from .base import Augmenter
import pandas as pd


class IdentityAugmenter(Augmenter):
    def __init__(self):
        super().__init__()
        
    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset
