from mopi.blocks.concat import DataSource

from mopi.blocks.models.random import AllOnesModel, RandomModel, AllZerosModel

from mopi.blocks.pipeline import Pipeline

from ..evaluation.classification import classification_metrics
from ..evaluation.calibration import calibration_metrics

from mopi.type import (
    Experiment,
    DatasetSplit,
)

from ..dataset.tweet_eval import get_tweet_eval_dataloader
from ..dataset.dynahate import get_dynahate_dataloader
from .utils import populate_experiments_with_pipelines


dataloader_tweeteval = get_tweet_eval_dataloader("hate")
global_dataloader = get_dynahate_dataloader()

tweet_eval_hate = DataSource("tweet_eval_hate", dataloader_tweeteval)

all_0s = Pipeline(
    "all_0s",
    tweet_eval_hate,
    [AllZerosModel("all_0s")],
)
all_1s = Pipeline(
    "all_1s",
    tweet_eval_hate,
    [AllOnesModel("all_1s")],
)
random = Pipeline(
    "random",
    tweet_eval_hate,
    [RandomModel("random")],
)

from .hate_speech import sklearn_simple_nb

pipelines_to_evaluate = [sklearn_simple_nb]  # [all_0s, all_1s, random]
metrics = classification_metrics + calibration_metrics

tweeteval_simple = [
    Experiment(
        project_name="hate-speech-detection-tweeteval",
        run_name="tweeteval",
        dataset_category=DatasetSplit.train,
        pipeline=all_0s,
        metrics=metrics,
        train=True,
        global_dataloader=global_dataloader,
    ),
    Experiment(
        project_name="hate-speech-detection-tweeteval",
        run_name="tweeteval",
        dataset_category=DatasetSplit.test,
        pipeline=all_0s,
        metrics=metrics,
        train=False,
        global_dataloader=global_dataloader,
    ),
]


all_experiments = populate_experiments_with_pipelines(
    tweeteval_simple, pipelines_to_evaluate
)
