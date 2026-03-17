from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model1 = LogisticRegression(max_iter=2000)
model2 = DecisionTreeClassifier(max_depth=5)

voting = VotingClassifier(
    estimators=[
        ("lr", model1),
        ("tree", model2),
    ],
    voting="soft"
)

voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)

print("accuracy:", accuracy_score(y_test, y_pred))

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

print("LR accuracy:", accuracy_score(y_test, model1.predict(X_test)))
print("Tree accuracy:", accuracy_score(y_test, model2.predict(X_test)))