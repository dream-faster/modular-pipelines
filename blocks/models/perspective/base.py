from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


from blocks.models.base import Model
from type import DataType, Evaluators, PredsWithProbs, SKLearnConfig
from ...io import PickleIO

from perspective import PerspectiveAPI
from utils.env_interface import get_env
import time
from utils.printing import PrintFormats, logger
import random


@dataclass
class PerspectiveConst:
    toxicity_score = "TOXICITY"


class PerspectiveModel(PickleIO, Model):

    inputTypes = [DataType.Series, DataType.List, DataType.NpArray]
    outputType = DataType.PredictionsWithProbs

    trained = False

    def __init__(self, id: str, evaluators: Optional[Evaluators] = None):
        self.id = id
        self.evaluators = evaluators
        self.trained = True

        self.model = PerspectiveAPI(get_env("PERSPECTIVE_TOKEN"))

    def predict(self, dataset: pd.Series) -> List[PredsWithProbs]:
        results = [self.get_rate_limited_result(text) for text in dataset]

        return results

    def is_fitted(self) -> bool:
        return self.trained

    def get_rate_limited_result(self, text: str) -> PredsWithProbs:
        try:
            score = self.model.score(text)[PerspectiveConst.toxicity_score]
            time.sleep(1.01)

            return (
                0 if score < 0.5 else 1,
                [1 - score, score],
            )
        except Exception as e:
            logger.log(
                f"Couldn't get a score for: \n'{PrintFormats.BOLD}{text}{PrintFormats.END}' \nRandomly choosing category.",
                level=logger.levels.THREE,
            )
            logger.log(e)

            return random.sample([(0, [1.0, 0.0]), (1, [0.0, 1.0])], k=1)[0]
