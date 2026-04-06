import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

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

split_index = int(len(feature_df) * 0.7)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

xgb_model = XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42,
)

xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)

xgb_comparison_df = feature_df.iloc[split_index:][["date", "sales"]].copy()
xgb_comparison_df["predicted"] = xgb_preds
xgb_comparison_df = xgb_comparison_df.rename(columns={"sales": "actual"})

xgb_mae = mean_absolute_error(y_test, xgb_preds)

ts = df.set_index("date")["sales"].asfreq("D")

test_start_date = feature_df.iloc[split_index]["date"]

arima_train = ts[ts.index < test_start_date]
arima_test = ts[ts.index >= test_start_date]

arima_model = ARIMA(arima_train, order=(1, 1, 1))
arima_model_fit = arima_model.fit()

arima_preds = arima_model_fit.forecast(steps=len(arima_test))

arima_mae = mean_absolute_error(arima_test, arima_preds)

print("XGBoost MAE:", xgb_mae)
print("ARIMA MAE:", arima_mae)

if xgb_mae < arima_mae:
    print("Winner: XGBoost")
else:
    print("Winner: ARIMA")