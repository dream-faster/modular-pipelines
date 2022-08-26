from typing import Callable, List

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from type import Evaluators, PredsWithProbs


def __wrap_sklearn_scorer(scorer: Callable, *args, **kwargs) -> Callable:
    def wrapper(y_true, predicted_probs: List[PredsWithProbs]) -> float:
        return scorer(y_true, [item[0] for item in predicted_probs], *args, **kwargs)

    return wrapper


classification_metrics: Evaluators = [
    ("f1_micro", __wrap_sklearn_scorer(f1_score, average="micro")),
    ("f1_macro", __wrap_sklearn_scorer(f1_score, average="macro")),
    ("f1_weighted", __wrap_sklearn_scorer(f1_score, average="weighted")),
    ("accuracy", __wrap_sklearn_scorer(accuracy_score)),
    ("precision", __wrap_sklearn_scorer(precision_score)),
    ("recall", __wrap_sklearn_scorer(recall_score)),
    ("roc_auc", __wrap_sklearn_scorer(roc_auc_score)),
    ("report", __wrap_sklearn_scorer(classification_report)),
]
