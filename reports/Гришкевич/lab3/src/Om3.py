
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

print("   Запуск задачи классификации (Breast Cancer Wisconsin)")

df = pd.read_csv("breast_cancer.csv")

df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

X = df.drop("diagnosis", axis=1)
y = (df["diagnosis"] == "M").astype(int)

print("Количество записей:", len(df))
print("Количество признаков:", X.shape[1])
print("Доброкачественные опухоли:", (y == 0).sum())
print("Злокачественные опухоли:", (y == 1).sum())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale")
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "Модель": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "False Negatives": fn
    })

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nМатрица ошибок:")
    print(cm)

    print("\nАнализ матрицы ошибок:")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    plt.figure(figsize=(6, 5))

    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues
    )

    plt.title(f"Матрица ошибок — {name}")
    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Доброкачественная", "Злокачественная"],
        rotation=15
    )

    plt.yticks(
        [0, 1],
        ["Доброкачественная", "Злокачественная"]
    )

    thresh = cm.max() / 2

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.xlabel("Предсказанный класс")
    plt.ylabel("Фактический класс")
    plt.tight_layout()
    plt.show()

results = pd.DataFrame(results)

print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
print(results.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

best_model = results.sort_values(
    by=["False Negatives", "Recall"],
    ascending=[True, False]
).iloc[0]

print("\nНАИБОЛЕЕ НАДЕЖНАЯ МОДЕЛЬ")
print(best_model["Модель"])
print("Количество ложноотрицательных прогнозов:", int(best_model["False Negatives"]))
print(f"Recall: {best_model['Recall']:.4f}")
