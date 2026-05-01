from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_DIR = "models/day143-distilbert-imdb/checkpoint-25"
LABELS = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE",
}
CONFIDENCE_THRESHOLD = 0.7

app = FastAPI(
    title="Transformer Sentiment API",
    description="A small API for serving a fine-tuned transformer sentiment model.",
    version="0.1.0",
)

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_DIR,
    tokenizer="distilbert-base-uncased",
)

class PredictRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=2000,
        description="Text for sentiment analysis",
    )

class PredictResponse(BaseModel):
    label: str
    score: float
    is_confident: bool


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest) -> PredictResponse:
    result = classifier(request.text)[0]
    raw_label = result["label"]
    score = float(result["score"])

    return PredictResponse(
        label=LABELS.get(raw_label, raw_label),
        score=result["score"],
        is_confident=score >= CONFIDENCE_THRESHOLD,
    )

@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "model_dir": MODEL_DIR,
        "base_model": "distilbert-base-uncased",
        "task": "sentiment-analysis",
        "labels": ["NEGATIVE", "POSITIVE"],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "known_limitation": "The model was fine-tuned on a small subset and may be biased toward NEGATIVE predictions.",
    }