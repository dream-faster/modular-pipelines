from datasets import load_dataset, Dataset, Features, Value, ClassLabel
from typing import Tuple
import pandas as pd
from config import GlobalPreprocessConfig


def shorten_datasets(
    data: Tuple[Dataset, Dataset, Dataset],
    preprocess_config: GlobalPreprocessConfig,
) -> Tuple[Dataset, Dataset, Dataset]:
    if preprocess_config.train_size != -1:
        train_dataset = data[0].select(
            [i for i in list(range(min(len(data[0]), preprocess_config.train_size)))]
        )
    else:
        train_dataset = data[0]

    if preprocess_config.val_size != -1:
        val_dataset = data[1].select(
            [i for i in list(range(min(len(data[1]), preprocess_config.val_size)))]
        )
    else:
        val_dataset = data[1]

    if preprocess_config.test_size != -1:
        test_dataset = data[2].select(
            [i for i in list(range(min(len(data[2]), preprocess_config.test_size)))]
        )
    else:
        test_dataset = data[2]

    return train_dataset, val_dataset, test_dataset


def load_data(from_huggingface: bool = False) -> Tuple[Dataset, Dataset, Dataset]:
    if from_huggingface:
        dataset = load_dataset("SetFit/sst5")
        train_dataset = dataset["train"].shuffle(seed=42)
        val_dataset = dataset["validation"].shuffle(seed=42)
        test_dataset = dataset["test"].shuffle(seed=42)

    else:
        df_train = pd.read_json("data/original/train.jsonl", lines=True)
        df_dev = pd.read_json("data/original/dev.jsonl", lines=True)
        df_test = pd.read_json("data/original/test.jsonl", lines=True)

        train_dataset = Dataset.from_pandas(
            df_train,
            features=Features({"text": Value("string"), "label": ClassLabel(5)}),
        )
        val_dataset = Dataset.from_pandas(
            df_dev, features=Features({"text": Value("string"), "label": ClassLabel(5)})
        )
        test_dataset = Dataset.from_pandas(
            df_test,
            features=Features({"text": Value("string"), "label": ClassLabel(5)}),
        )

    return train_dataset, val_dataset, test_dataset
