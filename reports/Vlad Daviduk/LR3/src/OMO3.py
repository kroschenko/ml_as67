from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def main():
    folder = Path(__file__).resolve().parent
    df = pd.read_csv(folder / "winequality-white.csv", sep=";")

    X = df.drop(columns="quality")
    y = (df["quality"] >= 7).astype(int)

    print("\nКлассы: 0 — обычное вино, 1 — хорошее вино")
    print(y.value_counts().sort_index().rename("Количество").to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nОбучающая выборка:", len(X_train))
    print("Тестовая выборка:", len(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    knn_search = GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
        ]),
        {"knn__n_neighbors": list(range(1, 26))},
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )

    knn_search.fit(X_train, y_train)
    best_k = knn_search.best_params_["knn__n_neighbors"]

    print("\nЛучшее k по кросс-валидации:", best_k)
    print(f"F1 на кросс-валидации: {knn_search.best_score_:.4f}")

    k_results = pd.DataFrame({
        "k": list(range(1, 26)),
        "mean_f1": knn_search.cv_results_["mean_test_score"],
        "std_f1": knn_search.cv_results_["std_test_score"]
    })

    print("\nВлияние количества соседей:")
    print(k_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    k_results.to_csv(folder / "knn_results.csv", index=False)

    models = {
        "k-NN": knn_search.best_estimator_,
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
        ])
    }

    results = []

    for name, model in models.items():
        if name != "k-NN":
            model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        results.append({
            "Модель": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1-score": f1_score(y_test, prediction, zero_division=0)
        })

        print(f"\n{name}:")
        print(classification_report(
            y_test,
            prediction,
            target_names=["Обычное вино", "Хорошее вино"],
            digits=4,
            zero_division=0
        ))

    results = pd.DataFrame(results).sort_values("F1-score", ascending=False)

    print("\nСравнение моделей на тестовой выборке:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    results.to_csv(
        folder / "model_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nЛучший алгоритм: {results.iloc[0]['Модель']}")


if __name__ == "__main__":
    main()