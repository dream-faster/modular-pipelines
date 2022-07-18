from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from type import Evaluators

classification_metrics: Evaluators = [
    ("f1", f1_score),
    ("accuracy", accuracy_score),
    ("precision", precision_score),
    ("recall", recall_score),
    ("roc_auc", roc_auc_score),
    ("report", classification_report),
]
