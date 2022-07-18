from typing import Dict
import pandas as pd
from blocks.pipeline import Pipeline
from .store import Store
from typing import List, Union
from .integrity import check_integrity
from pprint import pprint


def run_pipeline(
    pipeline: Pipeline,
    data: Dict[str, Union[pd.Series, List]],
    labels: pd.Series,
    train: bool,
) -> pd.DataFrame:

    print("🗼 Hierarchy of Models:")
    pprint(pipeline.children())

    print("🆔 Verifying pipeline integrity")
    if not check_integrity(pipeline):
        raise Exception("Pipeline integrity check failed")

    store = Store(data, labels)

    print("💈 Loading existing models")
    pipeline.load()

    print("📡 Looking for remote models")
    pipeline.load_remote()

    if train:
        print("🏋️ Training pipeline")
        pipeline.fit(store)

        print("📡 Uploading models")
        pipeline.save_remote()

    print("🔮 Predicting with pipeline")
    return pipeline.predict(store)


# def predict_pipeline(pipeline: Pipeline, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
