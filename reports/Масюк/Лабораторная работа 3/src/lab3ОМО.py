
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)
# 1. Загрузка набора данных
columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

file_path = r"C:\Users\alex0\OneDrive\Рабочий стол\Лабы3\pima-indians-diabetes.csv"

try:
    data = pd.read_csv(
        file_path,
        comment="#",
        header=None,
        names=columns
    )
except FileNotFoundError:
    print("Ошибка: файл датасета не найден!")
    input("Нажмите Enter для выхода...")
    raise SystemExit

print("=" * 60)
print("АНАЛИЗ НАБОРА ДАННЫХ PIMA INDIANS DIABETES")
print("=" * 60)

print("\nПервые пять строк:")
print(data.head())

print("\nРазмер набора данных:", data.shape)

# 2. Подготовка данных
# Замена нулевых значений на пропуски
columns_to_replace = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]
data[columns_to_replace] = data[columns_to_replace].replace(
    0, np.nan
)
# Разделение признаков и целевой переменной
X = data.drop("Outcome", axis=1)
y = data["Outcome"].astype(int)
print("\nКоличество пропущенных значений:")
print(X.isnull().sum())
print("\nРаспределение классов:")
print(y.value_counts())
# 3. Разделение данных на выборки
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
print("\nОбучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)
# 4. Создание моделей

# Модель k-ближайших соседей
knn = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

# Дерево решений
tree = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ))
])

# Метод опорных векторов
svm = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="linear",
        probability=True,
        random_state=42
    ))
])
models = {
    "k-NN": knn,
    "Decision Tree": tree,
    "SVM": svm
}
# 5. Обучение и оценка моделей
results = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print("МОДЕЛЬ:", name)
    print("=" * 60)

    # Обучение модели
    model.fit(X_train, y_train)

    # Предсказание классов
    predictions = model.predict(X_test)

    # Вероятности положительного класса
    probabilities = model.predict_proba(X_test)[:, 1]

    # Вычисление метрик
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, pos_label=1, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    auc = roc_auc_score(y_test, probabilities)

    # Сохранение результатов
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC AUC": auc
    }

    print("\nМетрики качества:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

    # Матрица ошибок
    print("\nМатрица ошибок:")
    print(confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ))

    # Отчёт классификации
    print("\nОтчёт классификации:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Нет диабета", "Есть диабет"],
        zero_division=0
    ))
# 6. Сравнение моделей
results_df = pd.DataFrame(results).T
print("\n" + "=" * 60)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)
print(results_df.round(4).to_string())
# 7. Выбор лучшей модели по Recall
best_model = results_df["Recall"].idxmax()
best_recall = results_df.loc[best_model, "Recall"]
print("\n" + "=" * 60)
print("ЛУЧШАЯ МОДЕЛЬ")
print("=" * 60)
print("Название модели:", best_model)
print("Recall:", round(best_recall, 4))
print(
    "\nЛучшая модель обнаруживает наибольшую долю "
    "положительных случаев диабета."
)
# 8. Построение ROC-кривых
fig, ax = plt.subplots(figsize=(9, 6))
for name, model in models.items():

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name,
        ax=ax
    )
ax.set_title("ROC-кривые моделей классификации")
ax.grid(True)
plt.tight_layout()
plt.show()
# 9. Сохранение результатов
results_df.to_csv(
    "results.csv",
    encoding="utf-8-sig"
)
print("\nРезультаты сохранены в файл results.csv")
input("\nНажмите Enter для завершения программы...")
