import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def main() -> None:
  X, y = load_breast_cancer(return_X_y=True)

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, stratify=y, random_state=42 
  )

  model = LogisticRegression(max_iter=2000)
  model.fit(X_train, y_train)

  y_pred_05 = model.predict(X_test)
  cm = confusion_matrix(y_test, y_pred_05)

  tn, fp, fn, tp = cm.ravel()

  y_proba = model.predict_proba(X_test)[:, 1]

  cost_fp = 5
  cost_fn = 100

  def cost_for_threshold(threshold: float) -> int:
     y_pred = (y_proba >= threshold).astype(int)
     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
     return int(fp * cost_fp + fn * cost_fn)

  thresholds = np.linspace(0.0, 1.0, 101)

  best_t = None
  best_cost = None

  for t in thresholds:
      c = cost_for_threshold(float(t))
      if best_cost is None or c < best_cost:
          best_cost = c
          best_t = float(t)

  print("best_threshold:", best_t, "best_cost:", best_cost)

if __name__ == "__main__":
    main()