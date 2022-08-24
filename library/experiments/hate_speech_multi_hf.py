from copy import deepcopy


from blocks.concat import DataSource, VectorConcat

from blocks.models.huggingface import HuggingfaceModel
from blocks.pipeline import Pipeline

from library.evaluation.classification import classification_metrics
from transformers.training_args import TrainingArguments
from type import (
    DatasetSplit,
    HuggingfaceConfig,
    Experiment,
    HFTaskTypes,
)
from blocks.models.sklearn import SKLearnModel
from blocks.adaptors.classification_output import ClassificationOutputAdaptor

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
    save_strategy="no",
    log_level="critical",
    report_to="none",
    optim="adamw_torch",
    logging_strategy="steps",
    evaluation_strategy="epoch",
    logging_steps=1,
    # eval_steps = 10
)


huggingface_base_config = HuggingfaceConfig(
    preferred_load_origin=None,  # LoadOrigin.local,
    pretrained_model="distilbert-base-uncased",
    user_name="semy",
    task_type=HFTaskTypes.sentiment_analysis,
    remote_name_override=None,
    save_remote=True,
    save=True,
    num_classes=2,
    val_size=0.1,
    frozen=True,
    training_args=huggingface_training_args,
)

huggingface_distil_bert_config = huggingface_base_config.set_attr(
    "pretrained_model", "distilbert-base-uncased"
)
huggingface_distilroberta_config = huggingface_base_config.set_attr(
    "pretrained_model", "distilroberta-base"
)


""" Data """


tweet_eval_hate = DataSource("tweet_eval_hate", get_tweet_eval_dataloader("hate"))


""" Pipelines"""

huggingface_baseline_distilbert = Pipeline(
    "nlp_hf_distilbert",
    tweet_eval_hate,
    [
        HuggingfaceModel("hf-model", huggingface_distil_bert_config),
        ClassificationOutputAdaptor(select=0),
    ],
)

huggingface_distilroberta = Pipeline(
    "nlp_hf_distilroberta-base",
    tweet_eval_hate,
    [
        HuggingfaceModel("distilroberta-base", huggingface_distilroberta_config),
        ClassificationOutputAdaptor(select=0),
    ],
)


full_pipeline = Pipeline(
    "nlp_hf_meta-model-pipeline",
    VectorConcat(
        "concat-source",
        [huggingface_distilroberta, huggingface_baseline_distilbert],
        tweet_eval_hate,
    ),
    [
        SKLearnModel("sklearn-meta-model", sklearn_config),
    ],
)

metrics = classification_metrics

""" Experiments """
multi_hf_run_experiments = [
    Experiment(
        project_name="hate-speech-detection-hf",
        run_name="hf-meta-model",
        dataset_category=DatasetSplit.train,
        pipeline=full_pipeline,
        metrics=metrics,
        train=True,
    ),
    Experiment(
        project_name="hate-speech-detection-hf",
        run_name="hf-meta-model",
        dataset_category=DatasetSplit.test,
        pipeline=full_pipeline,
        metrics=metrics,
        train=False,
    ),
]
