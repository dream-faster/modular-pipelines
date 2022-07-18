from .store import Store
from type import Evaluators
from typing import Union, List
import pandas as pd
from utils.json import dump_json
from pprint import pprint

def evaluate(
    predictions: Union[List, pd.Series], store: Store, evaluators: Evaluators, path: str
):
    stats = pd.Series(dtype="float64")

    for id, evaluator in evaluators:
        output = evaluator(store.get_labels(), predictions)
        if isinstance(output, (float, int, list)):
            stats[id] = output
        elif isinstance(output, dict):
            dump_json(output, path + "/" + id + ".json")

    pprint(stats)
    stats.to_csv(path + "/stats.csv")
