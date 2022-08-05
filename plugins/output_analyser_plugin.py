from pprint import pprint
from typing import List

from blocks.base import Block, DataSource, Element
from blocks.pipeline import Pipeline

from .base import Plugin
from runner.store import Store
from configs.constants import Const

from collections import Counter
import numpy as np
import random


class OutputAnalyserPlugin(Plugin):
    def __init__(self, num_examples: int = 10):
        self.num_examples = num_examples

    def on_run_end(self, pipeline: Pipeline, store: Store):
        input = store.get_data(Const.input_col)
        final_output = store.get_data(Const.final_output)
        original_labels = store.get_labels()

        if type(final_output) == np.ndarray:
            final_output = final_output.tolist()
        if type(original_labels) == np.ndarray:
            original_labels = original_labels.tolist()

        predictions = [output[0] for output in final_output]
        probabilities = [output[1] for output in final_output]

        final_output_freq = Counter(predictions)
        original_labels_freq = Counter(original_labels)

        spaceing = "    ┃    {:<16} {:<16} {:<16}"

        print("    ┃")
        print("    ┃ Frequencies")
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

        spaceing_example = "    ┃    {:<30} {:>16} {:>16} {:>16}"
        print("    ┃")
        print(f"    ┃ Sampeling {self.num_examples} Examples")
        print(
            spaceing_example.format(
                "input text", "final_output", "original_labels", "confidence"
            )
        )
        print(spaceing_example.format("-" * 50, "-" * 16, "-" * 16, "-" * 16))

        random_indecies = random.sample(range(len(input)), self.num_examples)
        for i in range(self.num_examples):
            rand_i = random_indecies[i]
            print(
                spaceing_example.format(
                    input[rand_i][:50] + " " * max(0, (50 - len(input[rand_i]))),
                    predictions[rand_i],
                    original_labels[rand_i],
                    f"{round(max(probabilities[rand_i]) * 100, 2)}%",
                )
            )

        return pipeline, store
