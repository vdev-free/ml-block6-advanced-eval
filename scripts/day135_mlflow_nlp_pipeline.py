import pandas as pd
import mlflow
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


EXPERIMENT_NAME = "nlp_baselines_day135"
DATA_PATH = "data/text_samples_cleaned.csv"

def run_experiment(df: pd.DataFrame, vectorizer_type: str) -> None:
    texts = df["text_clean"]
    labels = df["label"]

    test_size = 0.25
    random_state = 42

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    if vectorizer_type == "bow":
        vectorizer = CountVectorizer()
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError(f"Unsupported vectorizer_type: {vectorizer_type}")

    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    results_df = pd.DataFrame(
        {
            "text": X_test_texts.values,
            "true_label": y_test.values,
            "pred_label": preds,
        }
    )
    results_df["is_correct"] = results_df["true_label"] == results_df["pred_label"]
    errors_df = results_df[results_df["is_correct"] == False].copy()

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)

    results_path = artifact_dir / f"results_{vectorizer_type}.csv"
    results_df.to_csv(results_path, index=False)

    errors_path = artifact_dir / f"errors_only_{vectorizer_type}.csv"
    errors_df.to_csv(errors_path, index=False)

    summary_path = artifact_dir / f"summary_{vectorizer_type}.txt"
    summary_text = (
        f"vectorizer_type: {vectorizer_type}\n"
        f"model_type: logistic_regression\n"
        f"test_size: {test_size}\n"
        f"random_state: {random_state}\n"
        f"accuracy: {accuracy}\n"
        f"vocab_size: {len(vectorizer.vocabulary_)}\n"
        f"num_train_samples: {len(X_train_texts)}\n"
        f"num_test_samples: {len(X_test_texts)}\n"
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    with mlflow.start_run(run_name=f"{vectorizer_type}_logreg"):
        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("dataset_name", DATA_PATH)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("vocab_size", len(vectorizer.vocabulary_))
        mlflow.log_metric("num_train_samples", len(X_train_texts))
        mlflow.log_metric("num_test_samples", len(X_test_texts))

        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(errors_path))

    print()
    print(f"Run finished for: {vectorizer_type}")
    print(f"Accuracy: {accuracy}")
    print(f"Vocab size: {len(vectorizer.vocabulary_)}")
    print(f"Saved artifact: {results_path}")
    print(f"Saved summary : {summary_path}")
    print(f"Saved errors  : {errors_path}")

def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)

    print(df.head())
    print()
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Experiment:", EXPERIMENT_NAME)

    run_experiment(df, vectorizer_type="bow")
    run_experiment(df, vectorizer_type="tfidf")


if __name__ == "__main__":
    main()