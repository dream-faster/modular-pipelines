from typing import Tuple
from datasets.arrow_dataset import Dataset
import pandas as pd
from type import PreprocessConfig, TestDataset, TrainDataset
from configs.constants import Const
from sklearn.model_selection import train_test_split


def transform_hatespeech_detection_dataset(
    dataset: Dataset, config: PreprocessConfig
) -> Tuple[TrainDataset, TestDataset]:

    train_data, test_data = train_test_split(
        pd.DataFrame(dataset["train"]), test_size=0.2
    )
    df_train = train_data[: config.train_size]
    df_test = test_data[: config.test_size]

    cols_to_rename = {
        config.input_col: Const.input_col,
        config.label_col: Const.label_col,
    }

    df_train = df_train.rename(columns=cols_to_rename)
    df_test = df_test.rename(columns=cols_to_rename)

    return df_train, df_test
