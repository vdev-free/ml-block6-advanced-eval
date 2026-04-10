import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    df = pd.read_csv("data/text_samples_cleaned.csv")

    # print(df.head())
    # print()
    # print("Rows:", len(df))
    # print("Columns:", df.columns.tolist())

    texts = df["text_clean"]
    labels = df["label"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    # print()
    # print("Shape of X:", X.shape)
    # print("Vocabulary size:", len(vectorizer.vocabulary_))
    # print("First 10 features:", vectorizer.get_feature_names_out()[:10])

    X_dense = X.toarray()

    # print()
    # print("BoW matrix:")
    # print(X_dense)

    feature_names = vectorizer.get_feature_names_out()
    first_row = X_dense[0]

    # print()
    # print("First text:")
    # print(texts.iloc[0])
    # print()

    # print("Non-zero features for first text:")

    # for word, count in zip(feature_names, first_row):
    #     if count > 0:
    #         print(f"{word}: {count}")
    X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # print()
    # print("y_test:", list(y_test))
    # print("preds :", list(preds))
    # print("Accuracy:", accuracy_score(y_test, preds))

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    feature_weights = list(zip(feature_names, coefficients))
    feature_weights_sorted = sorted(feature_weights, key=lambda x: x[1])

    # print()
    # print("Top negative words:")
    # for word, weight in feature_weights_sorted[:5]:
    #     print(f"{word}: {weight:.3f}")

    # print()
    # print("Top positive words:")
    # for word, weight in feature_weights_sorted[-5:]:
    #      print(f"{word}: {weight:.3f}")

    new_texts = [
    "i love this",
    "bad quality and broken item",
    "very happy with delivery",
    "worst product ever",
    ]

    new_X = vectorizer.transform(new_texts)
    new_preds = model.predict(new_X)

    print()
    print("Predictions for new texts:")
    for text, pred in zip(new_texts, new_preds):
        print(f"{text!r} -> {pred}") 


if __name__ == "__main__":
    main()