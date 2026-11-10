
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

print("   Запуск задачи классификации (Digits)")

digits = load_digits()

X = digits.data
y = digits.target

print("Количество записей:", len(X))
print("Количество признаков:", X.shape[1])
print("Количество классов:", len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Обучающая выборка:", len(X_train))
print("Тестовая выборка:", len(X_test))

models = {
    "k-NN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale")
    )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        "Модель": name,
        "Accuracy": accuracy
    })

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=np.arange(10),
        target_names=[str(i) for i in range(10)],
        digits=4
    ))

results = pd.DataFrame(results).sort_values(
    by="Accuracy",
    ascending=False
)

print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
print(results.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

print("\nЛУЧШАЯ МОДЕЛЬ")
print(results.iloc[0]["Модель"])
print(f"Accuracy: {results.iloc[0]['Accuracy']:.4f}")
