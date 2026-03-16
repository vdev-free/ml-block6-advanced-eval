from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

X, y =load_breast_cancer(return_X_y=True)

model = LogisticRegression(max_iter=2000)

param_grid = {
    'C': [0.1, 1, 10],
    'solver': ["liblinear", "lbfgs"]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
)

grid.fit(X, y)

# print("best params:", grid.best_params_)
# print("best score:", grid.best_score_)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=3,
    cv=5,
    random_state=42,
)

random_search.fit(X, y)

print("random best params:", random_search.best_params_)
print("random best score:", random_search.best_score_)