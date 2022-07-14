from .base import Transformation
from configs.constants import Const
import pandas as pd
from utils.spacy import get_spacy


class SpacyTokenizer(Transformation):
    def preload(self):
        self.nlp = get_spacy()

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset[Const.input_col] = dataset[Const.input_col].apply(self.nlp)
        return dataset