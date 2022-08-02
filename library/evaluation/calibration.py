from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pycaleva import CalibrationEvaluator

from type import Evaluators, PredsWithProbs


def __calibration_report(y_true: List, predicted: List[PredsWithProbs]) -> dict:
    ce = CalibrationEvaluator(
        y_true,
        np.array([item[1] for item in predicted])[:, 1],
        outsample=True,
        n_groups="auto",
    )
    metrics = ce.metrics()
    return {
        "brier": metrics.brier,
        "ace": metrics.ace,
        "mce": metrics.mce,
        "awlc": metrics.awlc,
    }


def __calibration_plot(y_true: List, predicted: List[PredsWithProbs]):
    ce = CalibrationEvaluator(
        y_true,
        np.array([item[1] for item in predicted])[:, 1],
        outsample=True,
        n_groups="auto",
    )
    fig = ce.calibration_plot()
    plt.close(fig)
    return fig


# calibration_metrics: Evaluators = [
#     ("calibration", __calibration_report),
#     ("calibration_plot", __calibration_plot),
# ]

calibration_metrics = []
