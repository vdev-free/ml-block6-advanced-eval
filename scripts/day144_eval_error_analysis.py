import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
import csv
from pathlib import Path

MODEL_DIR = "models/day143-distilbert-imdb/checkpoint-25"
BASE_MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "imdb"
LABELS = {
    0: "NEGATIVE",
    1: "POSITIVE",
}

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

dataset = load_dataset(DATASET_NAME)

test_dataset = dataset["test"].shuffle(seed=42).select(range(50))

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_test = test_dataset.map(tokenize_batch, batched=True)

trainer = Trainer(model=model)

predictions_output = trainer.predict(tokenized_test)

predicted_labels = np.argmax(predictions_output.predictions, axis=1)
true_labels = predictions_output.label_ids

accuracy = np.mean(predicted_labels == true_labels)

error_indices = np.where(predicted_labels != true_labels)[0]

first_error_index = error_indices[0]

example = test_dataset[int(first_error_index)]

first_error_index = error_indices[0]

example = test_dataset[int(first_error_index)]

def clean_text(text: str) -> str:
    return text.replace("<br /><br />", " ").strip()

# print("\nFirst 5 errors:")

for error_index in error_indices[:5]:
    example = test_dataset[int(error_index)]

    true_label = LABELS[int(true_labels[error_index])]
    predicted_label = LABELS[int(predicted_labels[error_index])]

    # print("\n--- Error ---")
    # print("Index:", error_index)
    # print("True label:", true_label)
    # print("Predicted label:", predicted_label)
    # print("Text:", clean_text(example["text"])[:400])

false_negative_count = np.sum((true_labels == 1) & (predicted_labels == 0))
false_positive_count = np.sum((true_labels == 0) & (predicted_labels == 1))

error_examples = []

for error_index in error_indices:
    example = test_dataset[int(error_index)]

    error_examples.append(
        {
            "index": int(error_index),
            "true_label": LABELS[int(true_labels[error_index])],
            "predicted_label": LABELS[int(predicted_labels[error_index])],
            "text": clean_text(example["text"])[:500],
        }
    )

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

error_report_path = REPORTS_DIR / "day144_error_examples.csv"

with error_report_path.open("w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["index", "true_label", "predicted_label", "text"]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(error_examples)

print("\nSaved error report to:", error_report_path)