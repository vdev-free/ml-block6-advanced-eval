from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
import numpy as np
import evaluate
from pathlib import Path

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/day147_ecommerce_messages.csv"
SEED = 42

set_seed(SEED)

dataset = load_dataset("csv", data_files=DATA_PATH)

LABEL_NAMES = sorted(set(dataset["train"]["label"]))

label2id = {label: index for index, label in enumerate(LABEL_NAMES)}
id2label = {index: label for label, index in label2id.items()}

def encode_label(example):
    example["label"] = label2id[example["label"]]
    return example

encoded_dataset = dataset["train"].map(encode_label)

train_indices = []
test_indices = []

for label_id in range(len(LABEL_NAMES)):
    label_indices = [
        index
        for index, example in enumerate(encoded_dataset)
        if example["label"] == label_id
    ]

    train_indices.extend(label_indices[:4])
    test_indices.extend(label_indices[4:])

train_dataset = encoded_dataset.select(train_indices)
test_dataset = encoded_dataset.select(test_indices)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )

tokenized_train = train_dataset.map(tokenize_batch, batched=True)
tokenized_test = test_dataset.map(tokenize_batch, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_NAMES),
    label2id=label2id,
    id2label=id2label,
)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return accuracy_metric.compute(
        predictions=predictions,
        references=labels,
    )


training_args = TrainingArguments(
    output_dir="models/day147-ecommerce-message-classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_steps=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()

reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

report_path = reports_dir / "day147_eval_summary.txt"

with report_path.open("w", encoding="utf-8") as file:
    file.write("Day 147 — E-commerce Message Classifier Evaluation\n")
    file.write("=" * 55 + "\n\n")
    file.write(f"Model: {MODEL_NAME}\n")
    file.write(f"Seed: {SEED}\n")
    file.write(f"Train size: {len(train_dataset)}\n")
    file.write(f"Test size: {len(test_dataset)}\n")
    file.write(f"Labels: {LABEL_NAMES}\n\n")
    file.write(f"Eval accuracy: {eval_results['eval_accuracy']:.4f}\n")
    file.write(f"Eval loss: {eval_results['eval_loss']:.4f}\n\n")
    file.write("Known limitation:\n")
    file.write(
        "Dataset is very small: only 4 train examples per category. "
        "Accuracy is expected to be unstable and not production-ready.\n"
    )

print("Saved eval report to:", report_path)
