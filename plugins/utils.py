from blocks.base import DataSource
from constants import Const
from runner.store import Store
from typing import List, Callable, Union
import random
import numpy as np
from collections import Counter
from utils.printing import PrintFormats, multi_line_formatter
import pandas as pd
import re


def get_output_statistics(
    store: Store, datasource: DataSource, analysis_functions: List[Callable]
) -> List[pd.DataFrame]:
    input = datasource.deplate(store, [], False)
    final_output = store.get_data(Const.final_output)
    original_labels = datasource.get_labels()

    if type(final_output) == np.ndarray:
        final_output = final_output.tolist()
    if type(original_labels) == np.ndarray:
        original_labels = original_labels.tolist()

    predictions, probabilities = store.data_to_preds_probs(final_output)

    dfs = []
    for analysis_function in analysis_functions:
        dfs.append(
            analysis_function(
                store, input, original_labels, final_output, predictions, probabilities
            )
        )

    return dfs


def print_output_statistics(
    store: Store,
    input: List[Union[str, int, float]],
    original_labels: List[Union[str, int, float]],
    final_output: List[Union[int, float]],
    predictions: List[Union[int, float]],
    probabilities: List[float],
) -> pd.DataFrame:
    final_output_freq = Counter(predictions)
    original_labels_freq = Counter(original_labels)

    string_to_return = ""

    spaceing = "    ┃    {:<16} {:<16} {:<16}"
    string_to_return += f"{PrintFormats.BOLD}    ┃ Frequencies{PrintFormats.END}\n"
    string_to_return += (
        f'{spaceing.format("category", "final_output", "original_labels")}\n'
    )
    string_to_return += f'{spaceing.format("-" * 16, "-" * 16, "-" * 16)}\n'

    combined_dict = {
        key: [
            key,
            f"{value} ({round(value / len(final_output) * 100, 2)}%)",
            f"{original_labels_freq[key]} ({round(original_labels_freq[key] / len(original_labels) * 100, 2)}%)",
        ]
        for key, value in final_output_freq.items()
    }
    frequencies = pd.DataFrame(
        combined_dict, index=["category", "final_output", "original_labels"]
    ).transpose()

    return frequencies


def print_example_outputs(
    store: Store,
    input: List[Union[str, int, float]],
    original_labels: List[Union[str, int, float]],
    final_output: List[Union[int, float]],
    predictions: List[Union[int, float]],
    probabilities: List[float],
) -> pd.DataFrame:
    spaceing_example = "    ┃    {:<50} {:>16} {:>16} {:>16}"
    num_examples = 10

    string_to_return = f"{PrintFormats.BOLD}    ┃ Sampeling {num_examples} Examples{PrintFormats.END}\n"
    string_to_return += f'{spaceing_example.format("input text", "final_output", "original_labels", "confidence")}\n'
    string_to_return += (
        f'{spaceing_example.format("-" * 50, "-" * 16, "-" * 16, "-" * 16)}\n'
    )

    new_df = pd.DataFrame(
        [], columns=["input text", "final_output", "original_labels", "confidence"]
    )

    random_indecies = random.sample(range(len(input)), num_examples)
    for i in random_indecies:
        sliced_input = (
            input[i][:50]
            if type(input[i]) is str
            else f"Not a string type: {type(input[i])}."
        )

        new_df = pd.concat(
            [
                new_df,
                pd.DataFrame(
                    [
                        [
                            sliced_input + " " * max(0, (50 - len(sliced_input))),
                            predictions[i],
                            original_labels[i],
                            f"{round(max(probabilities[i]) * 100, 2)}%",
                        ]
                    ],
                    columns=[
                        "input text",
                        "final_output",
                        "original_labels",
                        "confidence",
                    ],
                ),
            ]
        )

    return new_df


def print_correlation_matrix(
    store: Store,
    input: List[Union[str, int, float]],
    original_labels: List[Union[str, int, float]],
    final_output: List[Union[int, float]],
    predictions: List[Union[int, float]],
    probabilities: List[float],
) -> pd.DataFrame:

    all_prediction_dict = store.get_all_predictions()

    return pd.DataFrame.from_dict(all_prediction_dict).corr()
