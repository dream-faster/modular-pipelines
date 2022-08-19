import pandas as pd
from datasets.arrow_dataset import Dataset

from configs.constants import Const
from type import PreprocessConfig, DatasetSplit
from data.dataloader import DataLoader


def transform_dataset(dataset: Dataset, config: PreprocessConfig) -> dict:

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


def get_tweet_eval_dataloader(name: str) -> DataLoader:
    return DataLoader(
        "tweet_eval",
        PreprocessConfig(
            train_size=-1,
            val_size=-1,
            test_size=-1,
            input_col="text",
            label_col="label",
        ),
        transform_dataset,
        name=name,
    )
