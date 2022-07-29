import os
from typing import List, Union

import pandas as pd
from matplotlib.figure import Figure

from type import Evaluators, PredsWithProbs
from utils.json import dump_json, dump_str
from utils.printing import pprint_indent

from .store import Store


def evaluate(
    predictions: List[PredsWithProbs], store: Store, evaluators: Evaluators, path: str
) -> pd.Series:
    stats = pd.Series(dtype="float64")
    if not os.path.isdir(path):
        os.makedirs(path)

    for id, evaluator in evaluators:
        output = evaluator(store.get_labels(), predictions)
        if isinstance(output, (float, int, list)):
            stats[id] = output
        elif isinstance(output, dict):
            for item in output.items():
                stats[item[0]] = item[1]
        elif isinstance(output, str):
            dump_str(output, path + "/" + id + ".txt")
        elif isinstance(output, Figure):
            output.savefig(path + "/" + id + ".png")
        else:
            raise Exception(f"Evaluator returned an invalid type: {type(output)}")

    pprint_indent(stats)
    stats.to_csv(path + "/stats.csv")

    return stats
