from typing import Tuple

import pandas as pd
from datasets.arrow_dataset import Dataset
from sklearn.model_selection import train_test_split

from configs.constants import Const
from type import PreprocessConfig, TestDataset, TrainDataset


def transform_hatespeech_offensive_dataset(
    dataset: Dataset, config: PreprocessConfig, test_set_ratio=0.2
) -> Tuple[TrainDataset, TestDataset]:

    train_data, test_data = train_test_split(
        pd.DataFrame(dataset["train"]), test_size=test_set_ratio
    )
    df_train = train_data[: config.train_size]
    df_test = test_data[: config.test_size]

    cols_to_rename = {
        config.input_col: Const.input_col,
        config.label_col: Const.label_col,
    }

    df_train = df_train.rename(columns=cols_to_rename)
    df_test = df_test.rename(columns=cols_to_rename)
    df_train[Const.label_col] = df_train[Const.label_col].apply(lambda x: 1 if x else 0)
    df_test[Const.label_col] = df_test[Const.label_col].apply(lambda x: 1 if x else 0)

    return df_train, df_test
