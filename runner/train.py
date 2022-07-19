from typing import List
from blocks.base import Block
import pandas as pd

from blocks.models.base import Model
from .store import Store
from .evaluation import evaluate


def train_predict(model: Block, dataset: pd.Series, store: Store):
    if not model.is_fitted() or model.config.force_fit:
        print(f"    ├ Training {model.id}, {model.__class__.__name__}")
        model.fit(dataset, store.get_labels())

        if model.config.save:
            model.save()

    return predict(model, dataset, store)


def predict(model: Block, dataset: pd.Series, store: Store) -> List:
    print(f"    ├ Predicting on {model.id}, {model.__class__.__name__}")
    output = model.predict(dataset)

    if (
        isinstance(model, Model)
        and hasattr(model, "evaluators")
        and model.evaluators is not None
    ):
        predictions = [pred[0] for pred in output]
        stats = evaluate(
            predictions, store, model.evaluators, f"{store.path}/{model.id}"
        )

        store.set_stats(model.id, stats)

    return output
