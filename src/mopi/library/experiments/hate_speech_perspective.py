from mopi.blocks.concat import DataSource

from mopi.blocks.pipeline import Pipeline

from mopi.library.evaluation.classification import classification_metrics
from mopi.library.evaluation.calibration import calibration_metrics
from mopi.type import (
    Experiment,
    DatasetSplit,
)
from ..dataset.tweet_eval import get_tweet_eval_dataloader
from mopi.blocks.models.perspective import PerspectiveModel


tweet_eval_hate = DataSource("tweet_eval_hate", get_tweet_eval_dataloader("hate"))

perspective_baseline_pipeline = Pipeline(
    "perspective_baseline",
    tweet_eval_hate,
    [PerspectiveModel("perspective_model")],
)

metrics = classification_metrics + calibration_metrics


perspective_experiments = [
    Experiment(
        project_name="hate-speech-detection-hf",
        run_name="perspective-baseline",
        dataset_category=DatasetSplit.test,
        pipeline=perspective_baseline_pipeline,
        metrics=metrics,
        train=False,
    ),
]
