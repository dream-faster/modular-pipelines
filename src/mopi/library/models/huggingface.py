from transformers.training_args import TrainingArguments
from mopi.type import HuggingfaceConfig, HFTaskTypes


huggingface_config = HuggingfaceConfig(
    preferred_load_origin=None,
    pretrained_model="distilbert-base-uncased",
    task_type=HFTaskTypes.sentiment_analysis,
    user_name="semy",
    save_remote=True,
    save=True,
    num_classes=2,
    val_size=0.1,
    frozen=False,
    remote_name_override=None,
    training_args=TrainingArguments(
        output_dir="",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="no",
        log_level="critical",
        report_to="none",
        optim="adamw_torch",
        logging_strategy="steps",
        evaluation_strategy="epoch",
        logging_steps=1,
        # eval_steps = 10
    ),
)
