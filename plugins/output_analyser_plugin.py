from pprint import pprint
from typing import Iterable, List

from blocks.base import Block, DataSource, Element
from blocks.pipeline import Pipeline

from .base import Plugin
from runner.store import Store
from constants import Const

from collections import Counter
import numpy as np
import random
from utils.printing import PrintFormats, multi_line_formatter
from typing import Any, Tuple, Callable, Union
import matplotlib.pyplot as plt
import pandas as pd


class OutputAnalyserPlugin(Plugin):
    def __init__(self, num_examples: int = 10):
        self.num_examples = num_examples
        self.analysis_functions: List[Callable] = [
            self.print_output_statistics,
            self.print_example_outputs,
            self.print_correlation_matrix,
        ]

    def on_predict_end(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        random_indecies = random.sample(range(len(last_output)), self.num_examples)

        prefix = "    ┃    "

        selected_outputs = [last_output[i] for i in random_indecies]

        if isinstance(selected_outputs[0], Iterable):
            for i in selected_outputs:
                print(prefix + str(i))
        else:
            print(prefix + str(selected_outputs))

        return store, last_output

    def on_run_end(self, pipeline: Pipeline, store: Store):
        input = pipeline.datasource.deplate(store, [], False)
        final_output = store.get_data(Const.final_output)
        original_labels = pipeline.datasource.get_labels()

        if type(final_output) == np.ndarray:
            final_output = final_output.tolist()
        if type(original_labels) == np.ndarray:
            original_labels = original_labels.tolist()

        predictions, probabilities = store.data_to_preds_probs(final_output)

        for analysis_function in self.analysis_functions:
            print("    ┃")
            analysis_function(
                store,
                input,
                original_labels,
                final_output,
                predictions,
                probabilities,
            )
            print("    ┃")

        return pipeline, store

    def print_output_statistics(
        self,
        store: Store,
        input: List[Union[str, int, float]],
        original_labels: List[Union[str, int, float]],
        final_output: List[Union[int, float]],
        predictions: List[Union[int, float]],
        probabilities: List[float],
    ) -> None:
        final_output_freq = Counter(predictions)
        original_labels_freq = Counter(original_labels)

        spaceing = "    ┃    {:<16} {:<16} {:<16}"
        print(f"{PrintFormats.BOLD}    ┃ Frequencies{PrintFormats.END}")
        print(spaceing.format("category", "final_output", "original_labels"))
        print(spaceing.format("-" * 16, "-" * 16, "-" * 16))

        for key, value in final_output_freq.items():
            print(
                spaceing.format(
                    key,
                    f"{value} ({round(value / len(final_output) * 100, 2)}%)",
                    f"{original_labels_freq[key]} ({round(original_labels_freq[key] / len(original_labels) * 100, 2)}%)"
                    if key in original_labels_freq
                    else "None (0%)",
                )
            )

    def print_example_outputs(
        self,
        store: Store,
        input: List[Union[str, int, float]],
        original_labels: List[Union[str, int, float]],
        final_output: List[Union[int, float]],
        predictions: List[Union[int, float]],
        probabilities: List[float],
    ) -> None:
        spaceing_example = "    ┃    {:<50} {:>16} {:>16} {:>16}"

        print(
            f"{PrintFormats.BOLD}    ┃ Sampeling {self.num_examples} Examples{PrintFormats.END}"
        )
        print(
            spaceing_example.format(
                "input text", "final_output", "original_labels", "confidence"
            )
        )
        print(spaceing_example.format("-" * 50, "-" * 16, "-" * 16, "-" * 16))

        random_indecies = random.sample(range(len(input)), self.num_examples)
        for i in random_indecies:
            sliced_input = (
                input[i][:50]
                if type(input[i]) is str
                else f"Not a string type: {type(input[i])}."
            )
            print(
                spaceing_example.format(
                    sliced_input + " " * max(0, (50 - len(sliced_input))),
                    predictions[i],
                    original_labels[i],
                    f"{round(max(probabilities[i]) * 100, 2)}%",
                )
            )

    def print_correlation_matrix(
        self,
        store: Store,
        input: List[Union[str, int, float]],
        original_labels: List[Union[str, int, float]],
        final_output: List[Union[int, float]],
        predictions: List[Union[int, float]],
        probabilities: List[float],
    ) -> None:
        print(f"{PrintFormats.BOLD}    ┃ Correlation Matrix {PrintFormats.END}")

        all_prediction_dict = store.get_all_predictions()

        print(
            multi_line_formatter(
                pd.DataFrame.from_dict(all_prediction_dict).corr().to_string(),
                level=2,
            )
        )
