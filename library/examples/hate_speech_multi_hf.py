from copy import deepcopy


from blocks.adaptors import ListOfListsToNumpy
from blocks.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from blocks.concat import DataSource, VectorConcat

from blocks.ensemble import Ensemble
from blocks.models.huggingface import HuggingfaceModel
from blocks.pipeline import Pipeline

from blocks.transformations import (
    Lemmatizer,
    SKLearnTransformation,
    SpacyTokenizer,
    TextStatisticTransformation,
)
from configs.constants import Const
from data.transformation import transform_dataset
from datasets.load import load_dataset
from library.evaluation.classification import classification_metrics
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from transformers import TrainingArguments
from type import (
    DatasetSplit,
    HuggingfaceConfig,
    LoadOrigin,
    PreprocessConfig,
    Experiment,
    SKLearnConfig,
    HFTaskTypes,
)
from blocks.models.sklearn import SKLearnModel
from blocks.adaptors.classification_output import ClassificationOutputAdaptor

from utils.flatten import remove_none
from data.dataloader import DataLoader

""" Models """
preprocess_config = PreprocessConfig(
    train_size=-1,
    val_size=-1,
    test_size=-1,
    input_col="text",
    label_col="label",
)

huggingface_training_args = TrainingArguments(
    output_dir="",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=0.01,
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=True,
    log_level="critical",
    report_to="none",
    optim="adamw_torch",
    logging_strategy="steps",
    evaluation_strategy="epoch",
    logging_steps=1,
    # eval_steps = 10
)


huggingface_base_config = HuggingfaceConfig(
    preferred_load_origin=LoadOrigin.local,
    pretrained_model="distilbert-base-uncased",
    user_name="semy",
    task_type=HFTaskTypes.sentiment_analysis,
    remote_name_override=None,
    save_remote=True,
    save=True,
    num_classes=2,
    val_size=0.1,
    force_fit=False,
    training_args=huggingface_training_args,
)


huggingface_distil_bert_config = deepcopy(huggingface_base_config)
huggingface_distil_bert_config.pretrained_model = "distilbert-base-uncased"

huggingface_distilroberta_config = deepcopy(huggingface_base_config)
huggingface_distilroberta_config.pretrained_model = "distilroberta-base"

sklearn_config = SKLearnConfig(
    force_fit=False,
    save=True,
    preferred_load_origin=LoadOrigin.local,
    classifier=VotingClassifier(
        estimators=[
            ("nb", MultinomialNB()),
            ("lg", LogisticRegression(random_state=Const.seed)),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100, max_depth=7, random_state=Const.seed
                ),
            ),
        ],
        voting="soft",
    ),
    one_vs_rest=False,
    save_remote=False,
)

""" Data """

input_data = DataSource("input")

dataloader = DataLoader("tweet_eval", preprocess_config, transform_dataset, "hate")


""" Pipelines"""

huggingface_baseline_distilbert = Pipeline(
    "nlp_hf_distilbert",
    input_data,
    remove_none(
        [
            HuggingfaceModel("hf-model", huggingface_distil_bert_config),
            ClassificationOutputAdaptor(select=0),
        ]
    ),
)

huggingface_distilroberta = Pipeline(
    "nlp_hf_distilroberta-base",
    input_data,
    remove_none(
        [
            HuggingfaceModel("distilroberta-base", huggingface_distilroberta_config),
            ClassificationOutputAdaptor(select=0),
        ]
    ),
)


full_pipeline = Pipeline(
    "nlp_hf_meta-model-pipeline",
    VectorConcat(
        "concat-source", [huggingface_distilroberta, huggingface_baseline_distilbert]
    ),
    remove_none(
        [
            SKLearnModel("sklearn-meta-model", sklearn_config),
        ]
    ),
)

metrics = classification_metrics

""" Experiments """
multi_hf_run_experiments = [
    Experiment(
        project_name="hate-speech-detection-hf",
        run_name="hf-meta-model",
        dataloader=dataloader,
        dataset_category=DatasetSplit.train,
        pipeline=full_pipeline,
        preprocessing_config=preprocess_config,
        metrics=metrics,
        train=True,
    ),
    Experiment(
        project_name="hate-speech-detection-hf",
        run_name="hf-meta-model",
        dataloader=dataloader,
        dataset_category=DatasetSplit.test,
        pipeline=full_pipeline,
        preprocessing_config=preprocess_config,
        metrics=metrics,
        train=False,
    ),
]
