from copy import deepcopy
from transformers import TrainingArguments
from blocks.pipeline import Pipeline
from blocks.models.huggingface import HuggingfaceModel

from blocks.models.sklearn import SKLearnModel
from configs.constants import Const
from library.evaluation import classification
from type import LoadOrigin, PreprocessConfig, HuggingfaceConfig, SKLearnConfig
from blocks.pipeline import Pipeline
from blocks.transformations import Lemmatizer, SpacyTokenizer
from blocks.data import DataSource
from blocks.ensemble import Ensemble
from blocks.augmenters.spelling_autocorrect import SpellAutocorrectAugmenter
from blocks.transformations import SKLearnTransformation, TextStatisticTransformation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    VotingClassifier,
)
from utils.flatten import remove_none
from sklearn.preprocessing import MinMaxScaler
from blocks.adaptors import ListOfListsToNumpy

from library.evaluation import classification_metrics

preprocess_config = PreprocessConfig(
    train_size=100,
    val_size=100,
    test_size=100,
    input_col="text",
    label_col="label",
)

huggingface_training_args = TrainingArguments(
    output_dir="",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
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
    repo_name="finetuning-tweeteval-hate-speech",
    save_remote=True,
    save=True,
    num_classes=2,
    val_size=0.1,
    force_fit=False,
    training_args=huggingface_training_args,
)


huggingface_distil_bert_config = deepcopy(huggingface_base_config)
huggingface_distil_bert_config.pretrained_model = "distilbert-base-uncased"

huggingface_byt5_config = deepcopy(huggingface_base_config)
huggingface_byt5_config.pretrained_model = "Narrativa/byt5-base-tweet-hate-detection"
huggingface_byt5_config.preferred_load_origin = LoadOrigin.pretrained

input_data = DataSource("input")


huggingface_baseline_distilbert = Pipeline(
    "nlp_hf_distilbert",
    input_data,
    remove_none(
        [
            HuggingfaceModel("hf-model", huggingface_distil_bert_config),
        ]
    ),
)

huggingface_byt5 = Pipeline(
    "nlp_hf_byt5",
    input_data,
    remove_none(
        [
            HuggingfaceModel("hf-model-byt5", huggingface_byt5_config),
        ]
    ),
)

ensemble_hf_multi_transformer = Ensemble(
    "ensemble_hf_multi_transformer",
    [huggingface_baseline_distilbert, huggingface_byt5],
)
