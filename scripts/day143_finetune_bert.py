from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer

MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "imdb"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = load_dataset(DATASET_NAME)

small_train_dataset = dataset["train"].shuffle(seed=42).select(range(200))
small_test_dataset = dataset["test"].shuffle(seed=42).select(range(50))

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_train = small_train_dataset.map(tokenize_batch, batched=True)
tokenized_test = small_test_dataset.map(tokenize_batch, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="../models/day143-distilbert-imdb",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10,
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

print("Eval results:", eval_results)