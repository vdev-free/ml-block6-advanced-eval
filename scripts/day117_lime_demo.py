from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=["class 0", "class 1"],
    mode="classification"
)

exp = explainer.explain_instance(
    X_test[1],
    model.predict_proba,
    num_features=5
)

print("Prediction:", model.predict_proba(X_test[1].reshape(1, -1)))
print("\nLIME explanation:")
for feature, value in exp.as_list():
    print(feature, "->", value)