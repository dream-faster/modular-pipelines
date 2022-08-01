from typing import Callable, List, Optional

import numpy as np
from datasets import Dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from configs.constants import Const
from type import HuggingfaceConfig


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
    training_args: TrainingArguments,
    train_data: Dataset,
    val_data: Dataset,
    config: HuggingfaceConfig,
    parent_path: str,
    id: str,
    trainer_callbacks: Optional[List[Callable]],
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=trainer_callbacks,
    )

    trainer.train()
    trainer.evaluate()

    return trainer
