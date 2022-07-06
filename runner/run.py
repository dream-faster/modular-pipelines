
from typing import Dict
import pandas as pd
from model.pipeline import Pipeline

def train_pipeline(pipeline: Pipeline, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    pipeline.fit(data)
    return data

# def predict_pipeline(pipeline: Pipeline, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
