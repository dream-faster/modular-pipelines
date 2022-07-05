from typing import Tuple
import pandas as pd
from type import GlobalPreprocessConfig
from configs.constants import DataConst


def load_data(
    folder: str, config: GlobalPreprocessConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_json(f"{folder}/train.jsonl", lines=True)[: config.train_size]
    df_val = pd.read_json(f"{folder}/val.jsonl", lines=True)[: config.val_size]
    df_test = pd.read_json(f"{folder}/test.jsonl", lines=True)[: config.test_size]

    df_train = pd.concat([df_train, df_val])

    df_train = df_train.rename(
        columns={
            config.input_name: DataConst.input_name,
            config.label_name: DataConst.label_name,
        }
    )
    df_test = df_test.rename(
        columns={
            config.input_name: DataConst.input_name,
            config.label_name: DataConst.label_name,
        }
    )

    return df_train, df_test
