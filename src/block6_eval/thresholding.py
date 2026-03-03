import numpy as np


def threshold_by_quota(y_proba: np.ndarray, quota: float) -> float:
    if not (0.0 < quota < 1.0):
        raise ValueError("quota must be in (0, 1)")

    return float(np.quantile(y_proba, 1.0 - quota))

def predict_with_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    return (y_proba >= threshold).astype(int)