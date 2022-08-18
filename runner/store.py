from typing import Dict, List, Union, Iterable, Tuple

import pandas as pd

from utils.random import random_string
import numpy as np


class Store:

    path: str

    def __init__(
        self, data: Dict[str, Union[List, pd.Series]], labels: pd.Series, path: str
    ):
        self.data = data
        self.labels = labels
        self.id = "store-" + random_string(5)
        self.path = path
        self.stats = dict()

    def get_data(self, id: str) -> pd.Series:
        return self.data[id].copy()

    def get_labels(self) -> pd.Series:
        return self.labels.copy()

    def set_data(self, id: str, data: Union[pd.Series, List]) -> None:
        self.data[id] = data

    def set_stats(self, id: str, stats: pd.Series) -> None:
        self.stats[id] = stats

    def get_all_stats(self) -> pd.DataFrame:
        df = pd.DataFrame([])

        for key, value in self.stats.items():
            df[key] = value

        return df

    def get_all_predictions(self) -> dict:
        outputs = dict()
        for block_name, output in self.data.items():
            if isinstance(output, (pd.Series, np.ndarray)):
                output = output.tolist()

            if isinstance(output, Iterable):
                if (
                    isinstance(output[0], Iterable)
                    and isinstance(output[0], str) is False
                ):
                    predictions, probabilities = self.data_to_preds_probs(output)
                    output = predictions

                if not isinstance(output[0], str):
                    outputs[block_name] = output

        return outputs

    @staticmethod
    def data_to_preds_probs(
        final_output: List[Union[List, Tuple]]
    ) -> Tuple[Union[int, float]]:

        predictions = [output[0] for output in final_output]
        probabilities = [output[1] for output in final_output]

        return predictions, probabilities
