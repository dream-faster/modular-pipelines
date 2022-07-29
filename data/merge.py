from typing import List

import pandas as pd


def merge_datasets(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(datasets, axis=0)
