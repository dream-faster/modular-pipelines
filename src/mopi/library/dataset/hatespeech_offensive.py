import pandas as pd
from datasets.arrow_dataset import Dataset
from sklearn.model_selection import train_test_split

from mopi.constants import Const
from mopi.type import PreprocessConfig, DatasetSplit
from mopi.data.dataloader import DataLoader, HuggingfaceDataLoader


def transform_hatespeech_offensive_dataset(
    dataset: Dataset, config: PreprocessConfig, test_set_ratio=0.2
) -> dict:
    dataset_pd = pd.DataFrame(dataset["train"])
    dataset_pd = dataset_pd[dataset_pd[config.label_col] != 1]

    train_data, test_data = train_test_split(dataset_pd, test_size=test_set_ratio)
    df_train = train_data[: config.train_size]
    df_test = test_data[: config.test_size]

    cols_to_rename = {
        config.input_col: Const.input_col,
        config.label_col: Const.label_col,
    }

    df_train = df_train.rename(columns=cols_to_rename)
    df_test = df_test.rename(columns=cols_to_rename)
    df_train[Const.label_col] = df_train[Const.label_col].apply(
        lambda x: 1 if x == 0 else 0
    )
    df_test[Const.label_col] = df_test[Const.label_col].apply(
        lambda x: 1 if x == 0 else 0
    )

    return {DatasetSplit.train.value: df_train, DatasetSplit.test.value: df_test}


def get_hate_speech_offensive_dataloader() -> DataLoader:
    return HuggingfaceDataLoader(
        "hate_speech_offensive",
        PreprocessConfig(
            train_size=-1,
            val_size=-1,
            test_size=-1,
            input_col="tweet",
            label_col="class",
        ),
        transform_hatespeech_offensive_dataset,
    )
