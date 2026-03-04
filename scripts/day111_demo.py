import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from block6_eval.thresholding import threshold_by_quota, predict_with_threshold
import matplotlib.pyplot as plt


def main() -> None:
    # 1) Дані (готовий набір зі sklearn)
    X, y = load_breast_cancer(return_X_y=True)

    # 2) Ділимо на train/test (щоб тест був чесний)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Тренуємо просту модель
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # 4) Отримуємо ймовірності на test
    y_proba = model.predict_proba(X_test)[:, 1]

    # 5) Бізнес-правило: можемо обробити тільки 20% (quota)
    quota = 0.2
    t = threshold_by_quota(y_proba, quota=quota)

    # 6) Перетворюємо ймовірності в 0/1
    y_pred = predict_with_threshold(y_proba, threshold=t)

    # 7) Дивимось, що вийшло
    cm = confusion_matrix(y_test, y_pred)

    print("quota:", quota)
    print("threshold:", t)
    print("predicted 1 rate:", float(np.mean(y_pred)))
    print("confusion_matrix:\n", cm)

    # --- ROC curve (малюємо і зберігаємо) ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig("artifacts/day111/roc.png")
    plt.close()

    # --- PR curve (малюємо і зберігаємо) ---
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.tight_layout()
    plt.savefig("artifacts/day111/pr.png")
    plt.close()


if __name__ == "__main__":
    main()