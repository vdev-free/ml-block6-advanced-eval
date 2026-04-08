import pandas as pd

from src.nlp.preprocess import preprocess_text


def main() -> None:
    df = pd.read_csv("data/text_samples.csv")

    df["text_clean"] = df["text"].apply(preprocess_text)
    df["text_clean_no_digits"] = df["text"].apply(
        lambda text: preprocess_text(text, remove_digits=True)
    )

    print("Original vs cleaned:")
    print(df[["text", "text_clean", "text_clean_no_digits"]].head())
    print()

    output_path = "data/text_samples_cleaned.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned file: {output_path}")


if __name__ == "__main__":
    main()