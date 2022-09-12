from typing import Callable, List, Optional

import numpy as np
from datasets.arrow_dataset import Dataset
from datasets import load_metric
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollatorWithPadding

from mopi.constants import Const
from mopi.type import HuggingfaceConfig


def compute_metrics(eval_pred):
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="micro")[
        "f1"
    ]
    return {"accuracy": accuracy, "f1": f1}


def run_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    training_args: TrainingArguments,
    train_data: Dataset,
    val_data: Dataset,
    trainer_callbacks: Optional[List[Callable]],
) -> Trainer:
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
