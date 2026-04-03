import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

df = pd.DataFrame({
    "date": pd.date_range(start="2025-04-01", periods=12, freq="D"),
    "sales": [100, 102, 101, 105, 107, 110, 108, 111, 115, 117, 116, 120],
})

tscv = TimeSeriesSplit(n_splits=3)

mae_scores = []
mape_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    last_train_value = train_df["sales"].iloc[-1]
    preds = [last_train_value] * len(test_df)

    mae = mean_absolute_error(test_df["sales"], preds)
    mae_scores.append(mae)

avg_mae = sum(mae_scores) / len(mae_scores)

for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    last_train_value = train_df["sales"].iloc[-1]
    preds = [last_train_value] * len(test_df)

    mape = mean_absolute_percentage_error(test_df["sales"], preds)
    mape_scores.append(mape)

    print(f"\nFOLD {fold}")
    print("fold MAPE:", mape, f"(~ {mape * 100:.2f}%)")

avg_mape = sum(mape_scores) / len(mape_scores)

print("\nMAPE scores by fold:", mape_scores)
print("Average backtest MAPE:", avg_mape, f"(~ {avg_mape * 100:.2f}%)")