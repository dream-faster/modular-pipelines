import pandas as pd
from .base import Transformation
from configs.constants import Const


class PredictionsToText(Transformation):
    def __init__(self):
        super().__init__()

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:

        agg_text = map(
            str,
            zip(
                map(str, dataset[Const.preds_col]),
                map(str, dataset[Const.probs_col]),
            ),
        )
        return pd.DataFrame({Const.input_col: agg_text})
