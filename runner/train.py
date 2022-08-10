from typing import Callable, List, Optional

import pandas as pd

from blocks.base import Block
from blocks.models.base import Model

from .evaluation import evaluate
from .store import Store


def train_predict(
    model: Block,
    dataset: pd.Series,
    store: Store,
):
    if not model.is_fitted() and model.config.frozen == False:
        print(f"    ┃  ├─── Block: {model.id}, {model.__class__.__name__}")
        model.fit(dataset, store.get_labels())

        if model.config.save:
            model.save()

    return predict(model, dataset, store)


def predict(model: Block, dataset: pd.Series, store: Store) -> List:
    print(f"    ┃  ├─── Block: {model.id}, {model.__class__.__name__}")

    output = model.predict(dataset)

    if (
        isinstance(model, Model)
        and hasattr(model, "evaluators")
        and model.evaluators is not None
    ):
        stats = evaluate(output, store, model.evaluators, f"{store.path}/{model.id}")

        store.set_stats(model.id, stats)

    return output
