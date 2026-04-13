import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main() -> None:
    df = pd.read_csv("data/text_samples_cleaned.csv")

    # print(df.head())
    # print()
    # print("Rows:", len(df))
    # print("Columns:", df.columns.tolist())

    texts = df["text_clean"]
    labels = df["label"]

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    results_df = pd.DataFrame(
        {
            "text": X_test_texts.values,
            "true_label": y_test.values,
            "pred_label": preds,
        }
    )

    # print()
    # print(results_df)

    results_df["is_correct"] = results_df["true_label"] == results_df["pred_label"]

    errors_df = results_df[results_df["is_correct"] == False]

    # print()
    # print("Only errors:")
    # print(errors_df)

    def guess_possible_reason(text: str) -> str:
        if "helpful" in text or "polite" in text or "support" in text:
            return "Positive wording may be too weak or rare in train data."
        return "Needs manual review."


    errors_df = errors_df.copy()
    errors_df["possible_reason"] = errors_df["text"].apply(guess_possible_reason)

    print()
    print("Errors with possible reasons:")
    print(errors_df)


if __name__ == "__main__":
    main()