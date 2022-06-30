import pandas as pd
from transformers import pipeline
from data.dataset import RawDataset
from typing import Tuple
from tqdm import tqdm
import torch as t
from datasets import load_metric, Dataset, Features, Value, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np

device = 0 if t.cuda.is_available() else -1

def load_data()->Tuple[RawDataset,RawDataset,RawDataset]:
    df_train = pd.read_json('data/original/train.jsonl', lines=True)
    df_dev = pd.read_json('data/original/dev.jsonl', lines=True)
    df_test = pd.read_json('data/original/test.jsonl', lines=True)
    
    train_dataset = Dataset.from_pandas(
        df_train,
        features=Features({"text": Value("string"), "label": ClassLabel(5)}),
    )
    val_dataset = Dataset.from_pandas(
        df_dev, features=Features({"text": Value("string"), "label": ClassLabel(5)})
    )
    test_dataset = Dataset.from_pandas(
        df_test,
        features=Features({"text": Value("string"), "label": ClassLabel(5)}),
        
    )
    
    # train_dataset = RawDataset(df_train)
    # val_dataset = RawDataset(df_dev)
    # test_dataset = RawDataset(df_test)

    return train_dataset, val_dataset, test_dataset



def run_pipeline():
    train_dataset, val_dataset, test_dataset = load_data()
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_dataset_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_dataset_val = val_dataset.map(tokenize_function, batched=True)
  
    train_dataset = tokenized_dataset_train.shuffle(seed=42)
    eval_dataset = tokenized_dataset_val.shuffle(seed=42)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=5
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    metric = load_metric("f1")

    training_args = TrainingArguments(
        "test_trainer",
        evaluation_strategy="steps",
        eval_steps=1,
        logging_steps=1,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=16,
        learning_rate=3e-5,
        prediction_loss_only=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
if __name__ == '__main__':
    run_pipeline()