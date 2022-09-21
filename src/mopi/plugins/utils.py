from mopi.blocks.base import DataSource
from mopi.constants import Const
from mopi.runner.store import Store
from typing import List, Callable, Union, Tuple, Optional
import random
import numpy as np
from collections import Counter
from mopi.utils.printing import PrintFormats, multi_line_formatter
import pandas as pd
import re
from mopi.utils.printing import logger


def get_output_statistics(
    store: Store,
    datasource: DataSource,
    analysis_functions: List[Tuple[str, Callable]],
    log_it: Optional[bool] = False,
) -> List[Tuple[str, pd.DataFrame]]:
    input = datasource.deplate(store, [], False)
    final_output = store.get_data(Const.final_output)
    original_labels = datasource.get_labels()

    if type(final_output) == np.ndarray:
        final_output = final_output.tolist()
    elif type(final_output) == pd.Series or type(final_output) == pd.DataFrame:
        final_output = final_output.to_list()
    if type(original_labels) == np.ndarray:
        original_labels = original_labels.tolist()
    elif type(original_labels) == pd.Series or type(original_labels) == pd.DataFrame:
        original_labels = original_labels.to_list()

    predictions, probabilities = store.data_to_preds_probs(final_output)

    dfs = []
    for func_name, analysis_function in analysis_functions:
        stat_df = analysis_function(
            store, input, original_labels, final_output, predictions, probabilities
        )
        dfs.append((func_name, stat_df))

        if log_it:
            logger.log(
                f"{logger.formats.BOLD}{func_name}{logger.formats.END}",
                level=logger.levels.TWO,
            )
            logger.log("", level=logger.levels.THREE)
            logger.log(
                stat_df.to_string(),
                level=logger.levels.THREE,
                mode=logger.modes.MULTILINE,
            )
            logger.log("", level=logger.levels.THREE)

    return dfs


def get_output_frequencies(
    store: Store,
    input: List[Union[str, int, float]],
    original_labels: List[Union[str, int, float]],
    final_output: List[Union[int, float]],
    predictions: List[Union[int, float]],
    probabilities: List[float],
) -> pd.DataFrame:

    final_output_freq = Counter(predictions)
    original_labels_freq = Counter(original_labels)

    df = pd.DataFrame([], columns=["category", "final_output", "original_labels"])

    df["category"] = pd.Series(list(final_output_freq.keys()))
    df["final_output"] = pd.Series(list(final_output_freq.values())).apply(
        lambda x: f"{x} ({round(x / len(final_output) * 100, 2)}%)"
    )
    df["original_labels"] = pd.Series(
        [
            f"{original_labels_freq[key]} ({round(original_labels_freq[key] / len(original_labels) * 100, 2)}%)"
            for key, value in final_output_freq.items()
        ]
    )

    return df


def get_example_outputs(
    store: Store,
    input: List[Union[str, int, float]],
    original_labels: List[Union[str, int, float]],
    final_output: List[Union[int, float]],
    predictions: List[Union[int, float]],
    probabilities: List[float],
) -> pd.DataFrame:
    num_examples = 20

    df = pd.DataFrame(
        [], columns=["input text", "final_output", "original_labels", "confidence"]
    )

    df["input text"] = pd.Series(input).apply(lambda x: x[: min(len(x), 35)])
    df["final_output"] = pd.Series(predictions)
    df["original_labels"] = pd.Series(original_labels)
    df["confidence"] = pd.Series(probabilities).apply(lambda x: round(max(x) * 100, 2))

    df = df.sample(min(len(input), num_examples))

    return df


def get_correlation_matrix(
    store: Store,
    input: List[Union[str, int, float]],
    original_labels: List[Union[str, int, float]],
    final_output: List[Union[int, float]],
    predictions: List[Union[int, float]],
    probabilities: List[float],
) -> pd.DataFrame:

    all_prediction_dict = store.get_all_predictions()

    return pd.DataFrame.from_dict(all_prediction_dict).corr()
