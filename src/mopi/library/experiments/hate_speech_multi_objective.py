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


class HFModels(Enum):
    distilbert_base_uncased = "distilbert-base-uncased"
    distilroberta_base = "distilroberta-base"


huggingface_base_config = HuggingfaceConfig(
    preferred_load_origin=None,  # LoadOrigin.local,
    pretrained_model=HFModels.distilbert_base_uncased.value,
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


huggingface_distil_bert_binary_config = huggingface_base_config


huggingface_distilbert_emotion_config = (
    huggingface_base_config.set_attr(
        "pretrained_model", HFModels.distilbert_base_uncased.value
    )
    .set_attr("num_classes", 6)
    .set_attr("task_type", HFTaskTypes.text_classification)
)

huggingface_distilbert_emoji_config = (
    huggingface_base_config.set_attr(
        "pretrained_model", HFModels.distilbert_base_uncased.value
    )
    .set_attr("num_classes", 6)
    .set_attr("task_type", HFTaskTypes.text_classification)
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

hf_distilbert_hate = Pipeline(
    "hf-hate",
    datasource=tweet_eval_hate,
    models=[
        HuggingfaceModel(
            "distilbert-binary",
            huggingface_distil_bert_binary_config,
            dict_lookup={"LABEL_0": 0, "LABEL_1": 1},
        ),
    ],
)

hf_distilbert_sentiment = Pipeline(
    "hf-sentiment",
    datasource=tweet_eval_sentiment,
    models=[
        HuggingfaceModel(
            "distilbert-binary",
            huggingface_distil_bert_binary_config,
            dict_lookup={"LABEL_0": 0, "LABEL_1": 1},
        ),
    ],
    datasource_predict=tweet_eval_hate,
)

hf_distilbert_emotion = Pipeline(
    "hf-emotion",
    datasource=tweet_eval_emotion,
    models=[
        HuggingfaceModel(
            "distilbert-multiclass",
            huggingface_distilbert_emotion_config,
            dict_lookup={
                "sadness": 0,
                "joy": 1,
                "anger": 2,
                "fear": 3,
                "love": 4,
                "surprise": 5,
            },
        ),
    ],
    datasource_predict=tweet_eval_hate,
)

hf_distilbert_emoji = Pipeline(
    "hf-emoji",
    datasource=tweet_eval_emoji,
    models=[
        HuggingfaceModel(
            "distilbert-multiclass",
            huggingface_distilbert_emoji_config,
            dict_lookup={
                "\u2764": 0,
                "\ud83d\ude0d": 1,
                "\ud83d\ude02": 2,
                "\ud83d\udc95": 3,
                "\ud83d\udd25": 4,
                "\ud83d\ude0a": 5,
                "\ud83d\ude0e": 6,
                "\u2728": 7,
                "\ud83d\udc99": 8,
                "\ud83d\ude18": 9,
                "\ud83d\udcf7": 10,
                "\ud83c\uddfa\ud83c\uddf8": 11,
                "\u2600": 12,
                "\ud83d\udc9c": 13,
                "\ud83d\ude09": 14,
                "\ud83d\udcaf": 15,
                "\ud83d\ude01": 16,
                "\ud83c\udf84": 17,
                "\ud83d\udcf8": 18,
                "\ud83d\ude1c": 19,
            },
        ),
    ],
    datasource_predict=tweet_eval_hate,
)

hf_distilbert_irony = Pipeline(
    "hf-irony",
    datasource=tweet_eval_irony,
    models=[
        HuggingfaceModel(
            "distilbert-binary",
            huggingface_distil_bert_binary_config,
            dict_lookup={"LABEL_0": 0, "LABEL_1": 1},
        ),
    ],
    datasource_predict=tweet_eval_hate,
)

hf_distilbert_offensive = Pipeline(
    "hf-offensive",
    datasource=tweet_eval_offensive,
    models=[
        HuggingfaceModel(
            "distilbert-binary",
            huggingface_distil_bert_binary_config,
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
            hf_distilbert_hate,
            hf_distilbert_sentiment,
            hf_distilbert_emotion,
            hf_distilbert_emoji,
            hf_distilbert_offensive,
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
