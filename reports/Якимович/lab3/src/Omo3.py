
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("   Запуск задачи классификации (Iris)")

iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

print("Количество записей:", len(X))
print("Количество признаков:", X.shape[1])
print("Количество классов:", len(np.unique(y)))

print("\nРаспределение классов:")
print(pd.Series(y).value_counts().sort_index().to_string())

print("\nПервые 5 записей:")
print(iris.frame.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nОбучающая выборка:", len(X_train))
print("Тестовая выборка:", len(X_test))

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

knn_search = GridSearchCV(
    Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ]),
    {"knn__n_neighbors": list(range(1, 21))},
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

knn_search.fit(X_train, y_train)

best_k = knn_search.best_params_["knn__n_neighbors"]

print("\nПОДБОР ОПТИМАЛЬНОГО K")
print("Лучшее количество соседей:", best_k)
print(f"Accuracy на кросс-валидации: {knn_search.best_score_:.4f}")

models = {
    "k-NN": knn_search.best_estimator_,
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        ))
    ])
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
