from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

class ClassifyMessageRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="Customer message from an e-commerce store",
    )


class ClassifyMessageResponse(BaseModel):
    category: str
    score: float
    is_confident: bool

MODEL_DIR = "models/day147-ecommerce-message-classifier/final"
CONFIDENCE_THRESHOLD = 0.7

classifier = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
)

app = FastAPI(
    title="AI E-commerce Message Classifier API",
    description="API for classifying e-commerce customer messages.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "model_dir": MODEL_DIR,
        "task": "ecommerce-message-classification",
        "categories": [
            "delivery_issue",
            "general_question",
            "payment_issue",
            "positive_feedback",
            "product_quality",
            "return_refund",
        ],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "known_limitation": (
            "The model was trained on a very small educational dataset "
            "and is not production-ready."
        ),
    }

@app.post("/classify-message", response_model=ClassifyMessageResponse)
def classify_message(request: ClassifyMessageRequest) -> ClassifyMessageResponse:
    result = classifier(request.message)[0]

    score = float(result["score"])
    category = str(result["label"])

    return ClassifyMessageResponse(
        category=category,
        score=score,
        is_confident=score >= CONFIDENCE_THRESHOLD,
    )