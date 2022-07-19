from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
)
import numpy as np
from datasets import load_metric, Dataset
from type import HuggingfaceConfig
from configs.constants import Const


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = load_f1.compute(predictions=predictions, references=labels, average="micro")[
        "f1"
    ]
    return {"accuracy": accuracy, "f1": f1}


def run_training_pipeline(
    train_data: Dataset,
    val_data: Dataset,
    config: HuggingfaceConfig,
    pipeline_id: str,
    id: str,
) -> Trainer:

    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model, num_labels=config.num_classes
    )
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

    def preprocess_function(examples):
        return tokenizer(examples[Const.input_col], truncation=True)

    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_val = val_data.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"{Const.output_pipelines_path}/{pipeline_id}/{id}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch" if config.save else "NO",
        push_to_hub=True,
        log_level="critical",
        report_to="none",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    return trainer
