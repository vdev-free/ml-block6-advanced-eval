from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model1 = LogisticRegression(max_iter=2000)
model2 = DecisionTreeClassifier(max_depth=5)
model3 = RandomForestClassifier(n_estimators=50, random_state=42)

stacking = StackingClassifier(
    estimators=[
        ("lr", model1),
        ("tree", model2),
        ("RFC", model3),
    ],
    final_estimator=LogisticRegression()
)

stacking.fit(X_train, y_train)

y_pred = stacking.predict(X_test)

print("Stacking accuracy:", accuracy_score(y_test, y_pred))