from enum import Enum

from mopi.blocks.concat import ClassificationOutputConcat, DataSource

from mopi.blocks.models.huggingface import HuggingfaceModel
from mopi.blocks.pipeline import Pipeline

from mopi.library.evaluation.classification import classification_metrics
from transformers.training_args import TrainingArguments
from mopi.type import (
    DatasetSplit,
    HuggingfaceConfig,
    Experiment,
    HFTaskTypes,
)
from mopi.blocks.models.sklearn import SKLearnModel

from ..dataset.tweet_eval import get_tweet_eval_dataloader
from ..models.sklearn_voting import sklearn_config

""" Models """

huggingface_training_args = TrainingArguments(
    output_dir="",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=0.01,
    weight_decay=0.01,
    save_strategy="epoch",
    log_level="critical",
    report_to="none",
    optim="adamw_torch",
    logging_strategy="steps",
    evaluation_strategy="epoch",
    logging_steps=1,
    # eval_steps = 10
)


huggingface_base_config = HuggingfaceConfig(
    preferred_load_origin=None,
    pretrained_model="vinai/bertweet-base",
    user_name="semy",
    task_type=HFTaskTypes.sentiment_analysis,
    remote_name_override=None,
    save_remote=True,
    save=True,
    num_classes=2,
    val_size=0.1,
    frozen=False,
    training_args=huggingface_training_args,
)


huggingface_binary_config = huggingface_base_config
huggingface_multiclass_config = huggingface_base_config.set_attr(
    "task_type", HFTaskTypes.text_classification
)


""" Data """

tweet_eval_hate = DataSource("tweet_eval_hate", get_tweet_eval_dataloader("hate"))
tweet_eval_sentiment = DataSource(
    "tweet_eval_sentiment", get_tweet_eval_dataloader("sentiment")
)
tweet_eval_emoji = DataSource("tweet_eval_emoji", get_tweet_eval_dataloader("emoji"))
tweet_eval_emotion = DataSource(
    "tweet_eval_emotion", get_tweet_eval_dataloader("emotion")
)
tweet_eval_irony = DataSource("tweet_eval_irony", get_tweet_eval_dataloader("irony"))
tweet_eval_offensive = DataSource(
    "tweet_eval_offensive", get_tweet_eval_dataloader("offensive")
)

""" Pipelines"""

pipeline_hate = Pipeline(
    "hf-hate",
    datasource=tweet_eval_hate,
    models=[
        HuggingfaceModel(
            "bert-binary",
            huggingface_binary_config,
            dict_lookup={"LABEL_0": 0, "LABEL_1": 1},
        ),
    ],
)

pipeline_sentiment = Pipeline(
    "hf-sentiment",
    datasource=tweet_eval_sentiment,
    models=[
        HuggingfaceModel(
            "bert-multiclass",
            huggingface_multiclass_config.set_attr("num_classes", 3),
            dict_lookup={"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},
        ),
    ],
    datasource_predict=tweet_eval_hate,
)

pipeline_emotion = Pipeline(
    "hf-emotion",
    datasource=tweet_eval_emotion,
    models=[
        HuggingfaceModel(
            "bert-multiclass",
            huggingface_multiclass_config.set_attr("num_classes", 6),
            dict_lookup={
                "LABEL_0": 0,
                "LABEL_1": 1,
                "LABEL_2": 2,
                "LABEL_3": 3,
                "LABEL_4": 4,
                "LABEL_5": 5,
            },
        ),
    ],
    datasource_predict=tweet_eval_hate,
)

pipeline_emoji = Pipeline(
    "hf-emoji",
    datasource=tweet_eval_emoji,
    models=[
        HuggingfaceModel(
            "bert-multiclass",
            huggingface_multiclass_config.set_attr("num_classes", 20),
            dict_lookup={
                "LABEL_0": 0,
                "LABEL_1": 1,
                "LABEL_2": 2,
                "LABEL_3": 3,
                "LABEL_4": 4,
                "LABEL_5": 5,
                "LABEL_6": 6,
                "LABEL_7": 7,
                "LABEL_8": 8,
                "LABEL_9": 9,
                "LABEL_10": 10,
                "LABEL_11": 11,
                "LABEL_12": 12,
                "LABEL_13": 13,
                "LABEL_14": 14,
                "LABEL_15": 15,
                "LABEL_16": 16,
                "LABEL_17": 17,
                "LABEL_18": 18,
                "LABEL_19": 19,
                "LABEL_20": 20,
            },
        ),
    ],
    datasource_predict=tweet_eval_hate,
)

pipeline_irony = Pipeline(
    "hf-irony",
    datasource=tweet_eval_irony,
    models=[
        HuggingfaceModel(
            "bert-binary",
            huggingface_binary_config,
            dict_lookup={"LABEL_0": 0, "LABEL_1": 1},
        ),
    ],
    datasource_predict=tweet_eval_hate,
)

pipeline_offensive = Pipeline(
    "hf-offensive",
    datasource=tweet_eval_offensive,
    models=[
        HuggingfaceModel(
            "bert-binary",
            huggingface_binary_config,
            dict_lookup={"LABEL_0": 0, "LABEL_1": 1},
        ),
    ],
    datasource_predict=tweet_eval_hate,
)

full_pipeline = Pipeline(
    "hf-multi-objective-pipeline",
    datasource=ClassificationOutputConcat(
        "concat-source",
        [
            pipeline_hate,
            pipeline_sentiment,
            pipeline_emotion,
            pipeline_emoji,
            pipeline_offensive,
        ],
        datasource_labels=tweet_eval_hate,
    ),
    models=[
        SKLearnModel("sklearn-meta-model", sklearn_config),
    ],
)

metrics = classification_metrics

""" Experiments """
multi_objective_experiments = [
    Experiment(
        project_name="hate-speech-detection-multiobjective",
        run_name="tweeteval",
        dataset_category=DatasetSplit.train,
        pipeline=full_pipeline,
        metrics=metrics,
        train=True,
    ),
    Experiment(
        project_name="hate-speech-detection-multiobjective",
        run_name="tweeteval",
        dataset_category=DatasetSplit.test,
        pipeline=full_pipeline,
        metrics=metrics,
        train=False,
    ),
]
