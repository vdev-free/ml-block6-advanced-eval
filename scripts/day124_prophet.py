import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

df = pd.DataFrame({
    "ds": pd.date_range(start="2025-03-01", periods=14, freq="D"),
    "y": [100, 102, 101, 105, 107, 110, 108, 111, 115, 117, 116, 120, 122, 121],
})

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=3)

forecast = model.predict(future)

# print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

train_df = df.iloc[:-3].copy()
test_df = df.iloc[-3:].copy()

train_model = Prophet()
train_model.fit(train_df)

train_future = train_model.make_future_dataframe(periods=3)

train_forecast = train_model.predict(train_future)

test_forecast = train_forecast[["ds", "yhat"]].tail(3)

comparison_df = test_df.merge(test_forecast, on="ds")

mae = mean_absolute_error(comparison_df["y"], comparison_df["yhat"])

print("Prophet MAE:", mae)
