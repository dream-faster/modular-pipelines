import pandas as pd
from datasets.arrow_dataset import Dataset

from mopi.constants import Const
from mopi.type import PreprocessConfig, DatasetSplit
from mopi.data.dataloader import DataLoader, HuggingfaceDataLoader


def transform_dynahate_dataset(dataset: Dataset, config: PreprocessConfig) -> dict:

    df_train = pd.DataFrame(dataset[DatasetSplit.train.value][: config.train_size])
    if DatasetSplit.val.value in dataset:
        df_val = pd.DataFrame(dataset[DatasetSplit.val.value][: config.val_size])
        df_train = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    df_test = pd.DataFrame(dataset[DatasetSplit.test.value][: config.test_size])

    cols_to_rename = {
        config.input_col: Const.input_col,
        config.label_col: Const.label_col,
    }

    df_train = df_train.rename(columns=cols_to_rename)
    df_test = df_test.rename(columns=cols_to_rename)

    return {DatasetSplit.train.value: df_train, DatasetSplit.test.value: df_test}


def get_dynahate_dataloader() -> DataLoader:
    return HuggingfaceDataLoader(
        "aps/dynahate",
        PreprocessConfig(
            train_size=-1,
            val_size=-1,
            test_size=-1,
            input_col="text",
            label_col="label",
        ),
        transform_dynahate_dataset,
    )
