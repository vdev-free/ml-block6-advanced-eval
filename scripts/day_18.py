import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from sklearn.calibration import calibration_curve

data = {
    "pages_viewed": [2, 3, 5, 7, 1, 4, 6, 8, 2, 9],
    "time_on_site": [20, 35, 50, 80, 10, 45, 60, 95, 15, 110],
    "bought":       [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
}

df = pd.DataFrame(data)

X = df[["pages_viewed", "time_on_site"]]
y = df["bought"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_prob_raw = model.predict_proba(X_test)[:, 1]

brier_raw = brier_score_loss(y_test, y_prob_raw)
logloss_raw = log_loss(y_test, y_prob_raw, labels=[0, 1])

platt_model = CalibratedClassifierCV(
    estimator=LogisticRegression(),
    method="sigmoid",
    cv=2
)

platt_model.fit(X_train, y_train)

y_prob_platt = platt_model.predict_proba(X_test)[:, 1]

brier_platt = brier_score_loss(y_test, y_prob_platt)
logloss_platt = log_loss(y_test, y_prob_platt, labels=[0, 1])

isotonic_model = CalibratedClassifierCV(
    estimator=LogisticRegression(),
    method="isotonic",
    cv=2
)

isotonic_model.fit(X_train, y_train)

y_prob_isotonic = isotonic_model.predict_proba(X_test)[:, 1]

brier_isotonic = brier_score_loss(y_test, y_prob_isotonic)
logloss_isotonic = log_loss(y_test, y_prob_isotonic, labels=[0, 1])

prob_true_raw, prob_pred_raw = calibration_curve(
    y_test, y_prob_raw, n_bins=3, strategy="uniform"
)

# plt.figure(figsize=(6, 6))
# plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
# plt.plot(prob_pred_raw, prob_true_raw, marker="o", label="Raw model")

# plt.xlabel("Mean predicted probability")
# plt.ylabel("Fraction of positives")
# plt.title("Reliability Diagram")
# plt.legend()


prob_true_raw, prob_pred_raw = calibration_curve(
    y_test, y_prob_raw, n_bins=3, strategy="uniform"
)

prob_true_platt, prob_pred_platt = calibration_curve(
    y_test, y_prob_platt, n_bins=3, strategy="uniform"
)

prob_true_isotonic, prob_pred_isotonic = calibration_curve(
    y_test, y_prob_isotonic, n_bins=3, strategy="uniform"
)

plt.figure(figsize=(7, 7))

plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.plot(prob_pred_raw, prob_true_raw, marker="o", label="Raw model")
plt.plot(prob_pred_platt, prob_true_platt, marker="o", label="Platt")
plt.plot(prob_pred_isotonic, prob_true_isotonic, marker="o", label="Isotonic")

plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Reliability Diagram: Raw vs Platt vs Isotonic")
plt.legend()
plt.show()