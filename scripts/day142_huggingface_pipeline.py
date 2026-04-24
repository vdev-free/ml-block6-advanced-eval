from transformers import pipeline

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
)

# text = "I love machine learning. It is difficult, but very interesting."

# result = classifier(text)

# print(result)

texts = [
    "I love machine learning. It is difficult, but very interesting.",
    "This course is terrible and boring.",
    "The model works fast and the result is useful.",
]

def analyze_sentiment(texts: list[str]) -> list[dict[str, float | str]]:
    return classifier(texts)

results = analyze_sentiment(texts)

for text, result in zip(texts, results):
    print(f"\nText: {text}")
    print(f"Label: {result['label']}")
    print(f"Score: {result['score']:.4f}")