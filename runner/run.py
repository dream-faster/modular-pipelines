from typing import Dict
import pandas as pd
from model.pipeline import Pipeline
from .store import Store

def train_pipeline(pipeline: Pipeline, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    store = Store(data)
    pipeline.preload()
    pipeline.fit(store)
    return pipeline.predict(store)


# def predict_pipeline(pipeline: Pipeline, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
