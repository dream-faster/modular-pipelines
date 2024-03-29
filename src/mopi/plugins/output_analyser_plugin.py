from mopi.utils.printing import logger
from pprint import pprint
from typing import Iterable, List

from mopi.blocks.base import Block, DataSource, Element
from mopi.blocks.pipeline import Pipeline

from .base import Plugin
from mopi.runner.store import Store
from mopi.constants import Const

from collections import Counter
import numpy as np
import random
from mopi.utils.printing import PrintFormats, multi_line_formatter
from typing import Any, Tuple, Callable, Union
import pandas as pd
from mopi.type import SourceTypes
from mopi.utils.list import flatten
from .utils import (
    get_output_frequencies,
    get_example_outputs,
    get_correlation_matrix,
    get_output_statistics,
)
from mopi.utils.printing import logger


class OutputAnalyserPlugin(Plugin):
    def __init__(self, num_examples: int = 10):
        self.num_examples = num_examples
        self.analysis_functions = [
            ("Output Statistics", get_output_frequencies),
            ("Example Outputs", get_example_outputs),
            ("Correlation Matrix", get_correlation_matrix),
        ]

    def on_predict_end(self, store: Store, last_output: Any) -> Tuple[Store, Any]:
        random_indecies = random.sample(
            range(len(last_output)), min(len(last_output), self.num_examples)
        )
        selected_outputs = [last_output[i] for i in random_indecies]

        if isinstance(selected_outputs[0], Iterable):
            for i in selected_outputs:
                logger.log(str(i), level=logger.levels.THREE)
        else:
            logger.log(str(selected_outputs), level=logger.levels.THREE)

        return store, last_output

    def on_run_end(self, pipeline: Pipeline, store: Store):
        all_datasources = [
            block
            for block in flatten(pipeline.children(SourceTypes.predict))
            if type(block) is DataSource
        ]

        _ = [
            get_output_statistics(
                store, datasource, self.analysis_functions, log_it=True
            )
            for datasource in all_datasources
        ]

        return pipeline, store
