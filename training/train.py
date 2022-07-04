from model.base import BaseModel
import pandas as pd

def train_model(
    model: BaseModel, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame
):
    if not model.is_fitted() or model.config.force_fit:
        model.fit(train_dataset, val_dataset)
