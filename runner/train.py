from model.base import Model
import pandas as pd
from .store import Store
from configs.constants import Const


def train_predict(model: Model, dataset: pd.DataFrame, store: Store):
    if not model.is_fitted() or model.config.force_fit:
        print(f"Training {model.id}")
        train_dataset = pd.DataFrame(
            {
                Const.label_col: store.get_labels(),
                Const.input_col: dataset[Const.input_col],
            }
        )
        model.fit(train_dataset)

    return model.predict(dataset)
