from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import shap

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

# print('model trained')

explainer = shap.Explainer(model, X_train)

shap_values = explainer(X_test[:1])

# print("base value:", shap_values.base_values)
# print("shap values:", shap_values.values)

features_name = load_breast_cancer().feature_names

for name, value in zip(features_name, shap_values.values[0]):
    print(name, '->', value)

