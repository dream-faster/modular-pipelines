from typing import Dict
import pandas as pd
from blocks.pipeline import Pipeline
from .store import Store
from typing import List, Union


def train_pipeline(
    pipeline: Pipeline, data: Dict[str, Union[pd.Series, List]], labels: pd.Series
) -> pd.DataFrame:
    store = Store(data, labels)
    pipeline.preload()
    print("| Training pipeline")
    pipeline.fit(store)
    print("| Predicting pipeline")
    return pipeline.predict(store)


# def predict_pipeline(pipeline: Pipeline, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
