from model.base import Model
import pandas as pd


def train_model(model: Model, train_dataset: pd.DataFrame):
    if not model.is_fitted() or model.config.force_fit:
        print(f"Training {model.id}")
        model.fit(train_dataset)
