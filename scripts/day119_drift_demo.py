import pandas as pd
from sklearn.metrics import accuracy_score

reference_df = pd.DataFrame(
    {
        "pages_viewed": [2, 3, 4, 3, 5, 4, 2, 3],
        "time_on_site": [20, 35, 40, 30, 55, 45, 25, 38],
        "is_mobile": [1, 1, 0, 1, 0, 1, 1, 0],
    }
)

current_df = pd.DataFrame(
    {
        "pages_viewed": [6, 7, 8, 7, 9, 8, 6, 7],
        "time_on_site": [70, 85, 95, 90, 120, 100, 75, 88],
        "is_mobile": [1, 1, 1, 1, 1, 1, 1, 1],
    }
)

# print("Reference data:")
# print(reference_df)

# print("\nCurrent data:")
# print(current_df)

def mean_drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], threshold: float = 0.30) -> pd.DataFrame:
    rows = []

    for col in columns:
        ref_mean = reference[col].mean()
        cur_mean = current[col].mean()

        if ref_mean == 0:
            relative_shift = 0.0
        else:
            relative_shift = (cur_mean - ref_mean) / ref_mean

        rows.append(
            {
                "feature": col,
                "reference_mean": round(ref_mean, 3),
                "current_mean": round(cur_mean, 3),
                "relative_shift": round(relative_shift, 3),
                "drift_flag": abs(relative_shift) >= threshold,
            }
        )

    return pd.DataFrame(rows)


feature_report = mean_drift_report(
    reference=reference_df,
    current=current_df,
    columns=["pages_viewed", "time_on_site", "is_mobile"],
    threshold=0.30,
)

# print("\n=== DATA DRIFT REPORT ===")
# print(feature_report)

has_data_drift = feature_report["drift_flag"].any()

# print("\n=== FINAL DATA DRIFT DECISION ===")
# print("Data drift detected:", has_data_drift)

old_window = pd.DataFrame(
    {
        "y_true": [0, 1, 0, 1, 1, 0],
        "y_pred": [0, 1, 0, 1, 1, 0],
    }
)

current_window_with_labels = pd.DataFrame(
    {
        "y_true": [0, 1, 0, 1, 1, 0],
        "y_pred": [1, 1, 0, 0, 0, 0],
    }
)

old_accuracy = accuracy_score(old_window["y_true"], old_window["y_pred"])
current_accuracy = accuracy_score(
    current_window_with_labels["y_true"],
    current_window_with_labels["y_pred"],
)

# print("\n=== CONCEPT DRIFT CHECK ===")
# print("Old accuracy:", old_accuracy)
# print("Current accuracy:", current_accuracy)

accuracy_drop = old_accuracy - current_accuracy
concept_drift_flag = accuracy_drop >= 0.20

# print("Accuracy drop:", accuracy_drop)
# print("Possible concept drift:", concept_drift_flag)

print("\n=== FINAL MONITORING SUMMARY ===")

if has_data_drift:
    print("Alert: data drift detected.")
else:
    print("No data drift alert.")

if concept_drift_flag:
    print("Alert: possible concept drift detected.")
else:
    print("No concept drift alert.")