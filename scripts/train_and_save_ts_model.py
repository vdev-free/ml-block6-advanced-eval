import os
import joblib
import pandas as pd
from xgboost import XGBRegressor

df = pd.DataFrame({
    "date": pd.date_range(start="2025-05-01", periods=21, freq="D"),
    "sales": [
        100, 102, 101, 105, 108, 115, 118,
        103, 105, 104, 109, 112, 119, 122,
        106, 108, 107, 111, 114, 121, 124,
    ],
})

df["lag_1"] = df["sales"].shift(1)
df["lag_7"] = df["sales"].shift(7)
df["rolling_mean_3"] = df["sales"].shift(1).rolling(window=3).mean()

feature_df = df.dropna().copy()

feature_cols = ["lag_1", "lag_7", "rolling_mean_3"]
X = feature_df[feature_cols]
y = feature_df["sales"]

model = XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42,
)

model.fit(X, y)

os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/xgb_ts_model.joblib")

print("Model saved to artifacts/xgb_ts_model.joblib")