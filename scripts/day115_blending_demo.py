from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

# розбиваємо на 3 частини
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

X_meta, X_test, y_meta, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# базові моделі
model1 = LogisticRegression(max_iter=2000)
model2 = DecisionTreeClassifier(max_depth=5)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# predictions для meta-model
pred1 = model1.predict_proba(X_meta)[:, 1]
pred2 = model2.predict_proba(X_meta)[:, 1]

X_meta_new = np.column_stack([pred1, pred2])

# meta-model
meta_model = LogisticRegression()
meta_model.fit(X_meta_new, y_meta)

# тест
test_pred1 = model1.predict_proba(X_test)[:, 1]
test_pred2 = model2.predict_proba(X_test)[:, 1]

X_test_new = np.column_stack([test_pred1, test_pred2])

final_pred = meta_model.predict(X_test_new)

print("Blending accuracy:", accuracy_score(y_test, final_pred))
