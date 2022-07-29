from typing import Tuple

import pandas as pd
from datasets.arrow_dataset import Dataset

from configs.constants import Const
from type import PreprocessConfig, TestDataset, TrainDataset


def transform_dataset(
    dataset: Dataset, config: PreprocessConfig
) -> Tuple[TrainDataset, TestDataset]:

    df_train = pd.DataFrame(dataset["train"][: config.train_size])
    if "validation" in dataset:
        df_val = pd.DataFrame(dataset["validation"][: config.val_size])
        df_train = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    df_test = pd.DataFrame(dataset["test"][: config.test_size])

    cols_to_rename = {
        config.input_col: Const.input_col,
        config.label_col: Const.label_col,
    }

    df_train = df_train.rename(columns=cols_to_rename)
    df_test = df_test.rename(columns=cols_to_rename)

    return df_train, df_test
