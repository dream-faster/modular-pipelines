from typing import List, Optional

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from mopi.blocks.models.base import Model
from mopi.type import DataType, PredsWithProbs


class VaderModel(Model):

    inputTypes = [DataType.Series, DataType.List, DataType.NpArray]
    outputType = DataType.PredictionsWithProbs

    def load(self):
        nltk.download("vader_lexicon")
        self.sid = SentimentIntensityAnalyzer()

    def fit(self, dataset: pd.Series, labels: Optional[pd.Series]) -> None:
        pass

    def predict(self, dataset: pd.Series) -> List[PredsWithProbs]:
        probabilities = np.array(
            [
                polarity_scores_to_probabilities(self.sid.polarity_scores(item))
                for item in dataset
            ]
        )
        predictions = np.argmax(probabilities, axis=-1)

        return list(zip(predictions, probabilities))

    def is_fitted(self) -> bool:
        return True

    def save(self) -> None:
        pass


def polarity_scores_to_probabilities(scores: dict) -> np.ndarray:
    score = scores["compound"] * 0.5 + 0.5
    return np.array([score, 1 - score])
