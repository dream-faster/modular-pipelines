from blocks.base import Block
import pandas as pd
from .store import Store


def train_predict(model: Block, dataset: pd.DataFrame, store: Store):
    if not model.is_fitted() or model.config.force_fit:
        print(f"    ├ Training {model.id}, {model.__class__.__name__}")
        model.fit(dataset, store.get_labels())
        model.save()

    return predict(model, dataset)



def predict(model: Block, dataset: pd.DataFrame) -> pd.DataFrame:
    print(f"    |- Predicting on {model.id}, {model.__class__.__name__}")

    return model.predict(dataset)
