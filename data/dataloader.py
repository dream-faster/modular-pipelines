from typing import Tuple
import pandas as pd
from type import PreprocessConfig, TestDataset, TrainDataset
from configs.constants import Const


def load_data(
    folder: str, config: PreprocessConfig
) -> Tuple[TrainDataset, TestDataset]:
    df_train = pd.read_json(f"{folder}/train.jsonl", lines=True)[: config.train_size]
    df_val = pd.read_json(f"{folder}/val.jsonl", lines=True)[: config.val_size]
    df_test = pd.read_json(f"{folder}/test.jsonl", lines=True)[: config.test_size]

    df_train = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    cols_to_rename = {
        config.input_col: Const.input_col,
        config.label_col: Const.label_col,
    }

    df_train = df_train.rename(columns=cols_to_rename)
    df_test = df_test.rename(columns=cols_to_rename)

    return df_train, df_test
