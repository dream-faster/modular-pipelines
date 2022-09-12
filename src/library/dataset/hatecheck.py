import pandas as pd
from datasets.arrow_dataset import Dataset

from constants import Const
from type import PreprocessConfig, DatasetSplit
from data.dataloader import DataLoader, HuggingfaceDataLoader


def transform_hatecheck_dataset(dataset: Dataset, config: PreprocessConfig) -> dict:

    df_train = pd.DataFrame()
    df_test = pd.DataFrame(dataset["test"][: config.test_size])

    cols_to_rename = {
        "test_case": Const.input_col,
        "label_gold": Const.label_col,
    }

    df_test = df_test.rename(columns=cols_to_rename)
    df_test = df_test[[Const.input_col, Const.label_col]]

    df_test[Const.label_col] = df_test[Const.label_col].apply(
        lambda x: 1 if x == "hateful" else 0
    )

    return {DatasetSplit.train.value: df_train, DatasetSplit.test.value: df_test}


def get_hatecheck_dataloader() -> DataLoader:
    return HuggingfaceDataLoader(
        "Paul/hatecheck",
        PreprocessConfig(
            train_size=-1,
            val_size=-1,
            test_size=-1,
            input_col="text",
            label_col="label",
        ),
        transform_hatecheck_dataset,
    )
