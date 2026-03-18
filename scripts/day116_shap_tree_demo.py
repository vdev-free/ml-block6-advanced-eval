from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

explainer = shap.Explainer(model)

shap_values = explainer(X_test[:1])

for name, value in zip(feature_names, shap_values.values[0]):
    print(name, "->", value[1])  

shap_values = explainer(X_test[:50])

mean_importance = np.mean(np.abs(shap_values.values[:, :, 1]), axis=0)

top_indices = np.argsort(mean_importance)[::-1]

print("\nTOP FEATURES:")
for i in top_indices[:10]:
    print(feature_names[i], "->", mean_importance[i])