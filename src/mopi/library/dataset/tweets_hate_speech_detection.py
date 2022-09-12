import pandas as pd
from datasets.arrow_dataset import Dataset
from sklearn.model_selection import train_test_split

from mopi.constants import Const
from mopi.type import PreprocessConfig, DatasetSplit
from mopi.data.dataloader import DataLoader, HuggingfaceDataLoader


def transform_hatespeech_detection_dataset(
    dataset: Dataset, config: PreprocessConfig, test_set_ratio=0.2
) -> dict:

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

    return {DatasetSplit.train.value: df_train, DatasetSplit.test.value: df_test}


def get_tweets_hate_speech_detection_dataloader() -> DataLoader:
    return HuggingfaceDataLoader(
        "tweets_hate_speech_detection",
        PreprocessConfig(
            train_size=-1,
            val_size=-1,
            test_size=-1,
            input_col="tweet",
            label_col="label",
        ),
        transform_hatespeech_detection_dataset,
    )
