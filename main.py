from transformers import (AutoTokenizer, DebertaV2ForSequenceClassification,
                          TrainingArguments, Trainer, AdamW,
                          DataCollatorWithPadding, DebertaV2ForSequenceClassification, BertForSequenceClassification)
import tensorflow as tf
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import evaluate

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset = load_dataset("csv", data_files={'train': 'files/Map.csv', 'test': 'files/validation.csv'})
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenization(example):
    return tokenizer(example["verse"], padding="max_length")


dataset = dataset.map(tokenization, batched=True)
dataset = dataset.remove_columns("verse")
dataset = dataset.rename_column("story", "labels")
dataset.set_format(columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=24)
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="chatgpt_rumo",
    learning_rate=2e-5,
    optim='adamw_torch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=25,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("result/")
