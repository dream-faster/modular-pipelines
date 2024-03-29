from typing import Callable, List, Optional

import pandas as pd

from mopi.blocks.base import Block
from mopi.blocks.models.base import Model

from .evaluation import evaluate
from .store import Store
from mopi.utils.printing import logger


def train_predict(
    model: Block,
    dataset: pd.Series,
    labels: pd.Series,
    store: Store,
):
    if not model.is_fitted() or model.config.frozen == False:
        logger.log(
            f"Block: {model.id}, {model.__class__.__name__}", level=logger.levels.TWO
        )
        model.fit(dataset, labels)

        if model.config.save:
            model.save()

    return predict(model, dataset, labels, store)


def predict(model: Block, dataset: pd.Series, labels: pd.Series, store: Store) -> List:
    logger.log(
        f"Block: {model.id}, {model.__class__.__name__}", level=logger.levels.TWO
    )

    output = model.predict(dataset)

    if (
        isinstance(model, Model)
        and hasattr(model, "evaluators")
        and model.evaluators is not None
    ):
        stats = evaluate(output, labels, model.evaluators, f"{store.path}/{model.id}")

        store.set_stats(model.id, stats)

    return output
