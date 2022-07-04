from typing import Tuple
import pandas as pd
from type import GlobalPreprocessConfig


def load_data(
    folder: str, preprocessing_config: GlobalPreprocessConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_json(f"{folder}/train.jsonl", lines=True)[
        : preprocessing_config.train_size
    ]
    df_val = pd.read_json(f"{folder}/val.jsonl", lines=True)[
        : preprocessing_config.val_size
    ]
    df_test = pd.read_json(f"{folder}/test.jsonl", lines=True)[
        : preprocessing_config.test_size
    ]

    return df_train, df_val, df_test
