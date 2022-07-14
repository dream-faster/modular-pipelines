from typing import Dict
import pandas as pd
from blocks.pipeline import Pipeline
from .store import Store
from typing import List, Union


def run_pipeline(
    pipeline: Pipeline,
    data: Dict[str, Union[pd.Series, List]],
    labels: pd.Series,
    train: bool,
) -> pd.DataFrame:

    store = Store(data, labels)
    pipeline.preload()

    print("| Loading existing models")
    pipeline.load()

    if train:
        print("| Training pipeline")
        pipeline.fit(store)

        print("| Saving models in pipeline")
        pipeline.save()

    print("| Predicting pipeline")
    return pipeline.predict(store)


# def predict_pipeline(pipeline: Pipeline, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
