from utils.printing import logger
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
from type import SourceTypes
from utils.list import flatten
from .utils import (
    print_output_statistics,
    print_example_outputs,
    print_correlation_matrix,
    get_output_statistics,
)


class OutputAnalyserPlugin(Plugin):
    def __init__(self, num_examples: int = 10):
        self.num_examples = num_examples
        self.analysis_functions: List[Tuple[str, Callable]] = [
            ("Output Statistics", print_output_statistics),
            ("Example Outputs", print_example_outputs),
            ("Correlation Matrix", print_correlation_matrix),
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
        all_datasources = [
            block
            for block in flatten(pipeline.children(SourceTypes.predict))
            if type(block) is DataSource
        ]
        for datasource in all_datasources:
            result_dfs = get_output_statistics(
                store,
                datasource,
                [analysis_function[1] for analysis_function in self.analysis_functions],
            )

            for i, df in enumerate(result_dfs):
                logger.log(
                    f"{logger.formats.BOLD}{self.analysis_functions[i][0]}{logger.formats.END}\n",
                    level=logger.levels.ONE,
                )
                logger.log(
                    df.to_string(),
                    level=logger.levels.THREE,
                    mode=logger.modes.MULTILINE,
                )
                logger.log("", level=logger.levels.ONE)

        return pipeline, store
