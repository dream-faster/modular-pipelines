from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, AutoModelForMaskedLM, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from data.dataloader import load_data

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")

   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}


def run_training():
   train_dataset, val_dataset, test_dataset = load_data(huggingface=False)

   tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

   def preprocess_function(examples):
      return tokenizer(examples["text"], truncation=True)
   
   tokenized_train = train_dataset.map(preprocess_function, batched=True)
   tokenized_test = test_dataset.map(preprocess_function, batched=True)

   data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

   repo_name = "finetuning-sentiment-model-sst"
   
   training_args = TrainingArguments(
      output_dir=repo_name,
      learning_rate=2e-5,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      num_train_epochs=2,
      weight_decay=0.01,
      save_strategy="epoch",
      push_to_hub=False,
   )
   
   trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_train,
      eval_dataset=tokenized_test,
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
   )

   trainer.train()
   trainer.evaluate()

if __name__ == "__main__":
   run_training()