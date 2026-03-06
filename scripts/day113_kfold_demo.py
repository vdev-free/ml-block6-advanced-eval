from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

model = LogisticRegression(max_iter=2000)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv)

print("stratified scores:", scores)
print("kfold mean:", np.mean(scores))