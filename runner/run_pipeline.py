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

    print("💈 Loading existing models")
    pipeline.load()

    print("📡 Looking for remote models")
    pipeline.load_remote()

    if train:
        print("🏋️ Training pipeline")
        pipeline.fit(store)

        # print("💽 Saving models in pipeline")
        # pipeline.save()

        print("📡 Uploading models")
        pipeline.save_remote()

    print("🔮 Predicting with pipeline")
    return pipeline.predict(store)


# def predict_pipeline(pipeline: Pipeline, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
