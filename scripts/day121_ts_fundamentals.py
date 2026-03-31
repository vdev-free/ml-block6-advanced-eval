import pandas as pd
from sklearn.model_selection import train_test_split

dates = pd.date_range(start="2025-01-01", periods=10, freq="D")
sales = [100, 120, 110, 130, 150, 160, 170, 165, 180, 190]

df = pd.DataFrame({
    "date": dates,
    "sales": sales,
})

# print(df)

# print("\nПерший продаж:", df.iloc[0]["sales"])
# print("Останній продаж:", df.iloc[-1]["sales"])

# print("\nПерші 3 рядки:")
# print(df.head(3))

# print("\nОстанні 3 рядки:")
# print(df.tail(3))

# random_train, random_test = train_test_split(
#     df, 
#     test_size=0.3,
#     shuffle=True,
#     random_state=42,
# )

# print("\nRANDOM TRAIN:")
# print(random_train.sort_values("date"))

# print("\nRANDOM TEST:")
# print(random_test.sort_values("date"))

split_index = int(len(df) * 0.7)

time_train = df.iloc[:split_index]
time_test = df.iloc[split_index:]

# print("\nTIME TRAIN:")
# print(time_train)

# print("\nTIME TEST:")
# print(time_test)

last_train_value = time_train["sales"].iloc[-1]

baseline_predictions = [last_train_value] * len(time_test)

# print("\nLAST TRAIN VALUE:", last_train_value)
# print("BASELINE PREDICTIONS:", baseline_predictions)
# print("REAL TEST VALUES:", time_test["sales"].tolist())

real_values = time_test["sales"].tolist()

absolute_errors = [
    abs(real - pred)
    for real, pred in zip(real_values, baseline_predictions)
]

mae = sum(absolute_errors) / len(absolute_errors)

# print("\nABSOLUTE ERRORS:", absolute_errors)
# print("BASELINE MAE:", mae)

df["lag_1"] = df["sales"].shift(1)
df["lag_2"] = df["sales"].shift(2)
df["lag_3"] = df["sales"].shift(3)

df["rolling_mean_3"] = df["sales"].rolling(window=3).mean()

df["rolling_mean_3_past"] = df["lag_1"].rolling(window=3).mean()

feature_df = df.dropna().copy()

# print(feature_df)

feature_cols = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_mean_3_past"]

X = feature_df[feature_cols]
y = feature_df["sales"]

print("X:")
print(X)

print("\ny:")
print(y)
