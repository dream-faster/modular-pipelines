from typing import Tuple
from datasets.arrow_dataset import Dataset
import pandas as pd
from type import PreprocessConfig, TestDataset, TrainDataset
from configs.constants import Const


def transform_dataset(
    dataset: Dataset, config: PreprocessConfig
) -> Tuple[TrainDataset, TestDataset]:

    df_train = pd.DataFrame(dataset["train"][: config.train_size])
    df_val = pd.DataFrame(dataset["validation"][: config.val_size])
    df_test = pd.DataFrame(dataset["test"][: config.test_size])

    df_train = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    cols_to_rename = {
        config.input_col: Const.input_col,
        config.label_col: Const.label_col,
    }

    df_train = df_train.rename(columns=cols_to_rename)
    df_test = df_test.rename(columns=cols_to_rename)

    return df_train, df_test
