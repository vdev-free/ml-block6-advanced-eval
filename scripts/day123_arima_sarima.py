import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

dates = pd.date_range(start="2025-01-01", periods=12, freq="D")
sales = [100, 102, 101, 105, 107, 110, 108, 111, 115, 117, 116, 120]

df = pd.DataFrame({
    "date": dates,
    "sales": sales,
})

ts = df.set_index("date")["sales"]

model = ARIMA(ts, order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=3)

train = ts.iloc[:-3]
test = ts.iloc[-3:]

train_model = ARIMA(train, order=(1, 1, 1))
train_model_fit = train_model.fit()

test_forecast = train_model_fit.forecast(steps=3)

comparison_df = pd.DataFrame({
    "actual": test,
    "predicted": test_forecast,
})

mae = mean_absolute_error(test, test_forecast)

seasonal_dates = pd.date_range(start="2025-02-01", periods=14, freq="D")

seasonal_sales = [50, 55, 60, 65, 70, 90, 95, 50, 55, 60, 65, 70, 90, 95]

seasonal_df = pd.DataFrame({
    "date": seasonal_dates,
    "sales": seasonal_sales,
})

seasonal_ts = seasonal_df.set_index("date")["sales"].asfreq("D")

seasonal_model = SARIMAX(
    seasonal_ts,
    order=(1, 0, 0),
    seasonal_order=(1, 0, 0, 7),
)

seasonal_model_fit = seasonal_model.fit()

seasonal_forecast = seasonal_model_fit.forecast(steps=7)

print("SARIMA forecast for next 7 days:")
print(seasonal_forecast)