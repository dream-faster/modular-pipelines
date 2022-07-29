from typing import Tuple

import pandas as pd
from datasets.arrow_dataset import Dataset

from configs.constants import Const
from type import PreprocessConfig, TestDataset, TrainDataset


def transform_hatecheck_dataset(
    dataset: Dataset, config: PreprocessConfig
) -> Tuple[TrainDataset, TestDataset]:

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

    return df_train, df_test
