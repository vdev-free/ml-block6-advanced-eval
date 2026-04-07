from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="TS Forecast API")

model = joblib.load("artifacts/xgb_ts_model.joblib")


class PredictRequest(BaseModel):
    lag_1: float
    lag_7: float
    rolling_mean_3: float


class PredictResponse(BaseModel):
    prediction: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "TS Forecast API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    features = pd.DataFrame(
        [
            {
                "lag_1": payload.lag_1,
                "lag_7": payload.lag_7,
                "rolling_mean_3": payload.rolling_mean_3,
            }
        ]
    )

    prediction = model.predict(features)[0]

    return PredictResponse(prediction=float(prediction))