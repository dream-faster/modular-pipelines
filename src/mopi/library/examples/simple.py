#%%
# 1. Define your first pipeline
## Pipelines are build of a single Datasource and a list of Blocks (Models, Transformers and/or Augmenters)
## DataSources receive a dataloader, that either load data from remote or from a local folder (Here we load tweeteval using huggingface)
from mopi.blocks.pipeline import Pipeline
from mopi.blocks.concat import DataSource
from mopi.blocks.models.random import RandomModel
from mopi.library.dataset.tweet_eval import get_tweet_eval_dataloader

datasource = DataSource("tweet_eval_hate", get_tweet_eval_dataloader("hate"))
pipeline_blocks = [RandomModel("random_model_01")]
pipeline_name = "random_pipeline_name"


simple_pipeline = Pipeline(pipeline_name, datasource, pipeline_blocks)

#%%
# 2. Define experiments
## Experiments wrap your pipeline into reproducible objects that both train and test
from mopi.type import Experiment, DatasetSplit

## Define metrics to evaluate the performance of your model on.
from mopi.library.evaluation.classification import classification_metrics


all_experiments = [
    Experiment(
        project_name="simple-example",
        run_name="tweeteval",
        dataset_category=DatasetSplit.train,
        pipeline=simple_pipeline,
        metrics=classification_metrics,
        train=False,
    ),
    Experiment(
        project_name="simple-example",
        run_name="tweeteval",
        dataset_category=DatasetSplit.test,
        pipeline=simple_pipeline,
        metrics=classification_metrics,
        train=False,
    ),
]

# %%
