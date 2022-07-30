from copy import deepcopy

from blocks.data import DataSource
from blocks.ensemble import Ensemble
from blocks.models.huggingface import HuggingfaceModel
from blocks.pipeline import Pipeline
from transformers.training_args import TrainingArguments
from type import HuggingfaceConfig, LoadOrigin, PreprocessConfig
from utils.flatten import remove_none

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

huggingface_distilroberta_config = deepcopy(huggingface_base_config)
huggingface_distilroberta_config.pretrained_model = "distilroberta-base"
huggingface_distilroberta_config.preferred_load_origin = LoadOrigin.pretrained

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

huggingface_distilroberta = Pipeline(
    "nlp_hf_distilroberta-base",
    input_data,
    remove_none(
        [
            HuggingfaceModel("distilroberta-base", huggingface_distilroberta_config),
        ]
    ),
)

ensemble_hf_multi_transformer = Ensemble(
    "ensemble_hf_multi_transformer",
    [huggingface_baseline_distilbert, huggingface_distilroberta],
)
