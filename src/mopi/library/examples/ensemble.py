#%%
# 1. Define your first pipeline
## Pipelines are build of a single Datasource and a list of Blocks (Models, Transformers and/or Augmenters)
## DataSources receive a dataloader, that either load data from remote or from a local folder (Here we load tweeteval using huggingface)
### Import Pipeline and Datasource
from mopi.blocks.pipeline import Pipeline
from mopi.blocks.concat import DataSource
from mopi.library.dataset.tweet_eval import get_tweet_eval_dataloader

### Import Models
from mopi.blocks.models.sklearn import SKLearnModel
from mopi.blocks.models.vader import VaderModel
from mopi.blocks.models.huggingface import HuggingfaceModel
from mopi.library.models.sklearn_voting import sklearn_config
from mopi.library.models.huggingface import huggingface_config
from sklearn.feature_extraction.text import TfidfVectorizer
from mopi.utils.list import remove_none
from mopi.blocks.transformations import (
    Lemmatizer,
    SKLearnTransformation,
    SpacyTokenizer,
)


datasource = DataSource("tweet_eval_hate", get_tweet_eval_dataloader("hate"))


vader = Pipeline("vader", datasource, [VaderModel("vader")])
sklearn = Pipeline(
    "sklearn",
    datasource,
    remove_none(
        [
            SpacyTokenizer(),
            Lemmatizer(remove_stopwords=True),
            SKLearnTransformation(
                TfidfVectorizer(
                    max_features=100000,
                    min_df=2,
                    ngram_range=(1, 3),
                )
            ),
            SKLearnModel("sklearn", sklearn_config),
        ]
    ),
)

huggingface_baseline = Pipeline(
    "hf-distillbert",
    datasource,
    remove_none(
        [
            HuggingfaceModel(
                "hf-model", huggingface_config, dict_lookup={"LABEL_0": 0, "LABEL_1": 1}
            ),
        ]
    ),
)

#%%
# 2. Create Ensemble of previous pipelines
from mopi.blocks.ensemble import Ensemble

ensemble_sklearn_hf_vader = Ensemble(
    "ensemble_sklearn_hf_vader", datasource, [sklearn, vader, huggingface_baseline]
)


#%%
# 3. Define experiments
## Experiments wrap your pipeline into reproducible objects that both train and test
from mopi.type import Experiment, DatasetSplit

## Define metrics to evaluate the performance of your model on.
from mopi.library.evaluation.classification import classification_metrics


all_experiments = [
    Experiment(
        project_name="simple-example",
        run_name="tweeteval",
        dataset_category=DatasetSplit.train,
        pipeline=ensemble_sklearn_hf_vader,
        metrics=classification_metrics,
        train=False,
    ),
    Experiment(
        project_name="simple-example",
        run_name="tweeteval",
        dataset_category=DatasetSplit.test,
        pipeline=ensemble_sklearn_hf_vader,
        metrics=classification_metrics,
        train=False,
    ),
]

# %%
# 4. Run Experiments
from mopi.run import run
from mopi.type import Experiment, StagingConfig, StagingNames

prod_config = StagingConfig(
    name=StagingNames.prod,
    save_remote=False,
    log_remote=True,
    limit_dataset_to=None,
)

run(
    all_experiments,
    staging_config=prod_config,
)
