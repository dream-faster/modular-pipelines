from copy import deepcopy
from typing import List

from mopi.blocks.pipeline import Pipeline

from mopi.type import Experiment
from mopi.utils.list import flatten


def _set_pipeline(experiment: Experiment, pipeline: Pipeline) -> Experiment:
    experiment.pipeline = pipeline
    return experiment


def populate_experiments_with_pipelines(
    experiments: List[Experiment], pipelines: List[Pipeline]
) -> List[Experiment]:
    return flatten(
        [
            [
                [
                    _set_pipeline(deepcopy(experiment), pipeline)
                    for experiment in experiments
                ]
            ]
            for pipeline in pipelines
        ]
    )
