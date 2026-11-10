
from pathlib import Path
import sys
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# Папка, в которой находятся CSV-файлы
DATA_DIR = Path(r"C:\Users\user\Desktop")


# Названия столбцов исходного датасета Mushroom
MUSHROOM_COLUMNS = [
    "class", "cap-shape", "cap-surface", "cap-color",
    "bruises", "odor", "gill-attachment", "gill-spacing",
    "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color",
    "population", "habitat"
]


def find_csv(keyword):
    """Ищет CSV-файл на Рабочем столе по ключевому слову."""
    files = list(DATA_DIR.glob("*.csv"))

    for file in files:
        if keyword in file.stem.lower():
            return file

    raise FileNotFoundError(
        f"Не найден CSV-файл со словом '{keyword}' в названии.\n"
        f"Проверь папку: {DATA_DIR}"
    )


def read_csv_auto(path):
    """Читает CSV, автоматически определяя разделитель."""
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin1")


def load_student_data():
    """Загружает данные об успеваемости студентов."""
    path = find_csv("student")
    print(f"\nФайл студентов: {path.name}")

    data = read_csv_auto(path)

    # Удаляем пробелы из названий столбцов
    data.columns = data.columns.astype(str).str.strip()

    # Приводим названия к нижнему регистру
    data.columns = [column.lower() for column in data.columns]

    required = ["studytime", "failures", "g1", "g2", "g3"]

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            f"В файле студентов отсутствуют столбцы: {missing}"
        )

    # Оставляем только нужные для лабораторной столбцы
    data = data[required].copy()

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna()

    if len(data) < 10:
        raise ValueError("В файле студентов слишком мало корректных строк.")

    return data


def load_mushroom_data():
    """Загружает данные о съедобных и ядовитых грибах."""
    path = find_csv("mushroom")
    print(f"\nФайл грибов: {path.name}")

    data = read_csv_auto(path)

    # Если файл содержит исходные 23 столбца UCI без заголовков
    if len(data.columns) == 23:
        first = str(data.columns[0]).strip().lower()

        if first not in ("class", "label", "target"):
            data.columns = MUSHROOM_COLUMNS

    # Нормализуем заголовки
    data.columns = data.columns.astype(str).str.strip().str.lower()

    # Удаляем случайный столбец индекса, если он есть
    data = data.loc[
        :,
        ~data.columns.str.match(r"^unnamed")
    ]

    # Определяем целевой столбец
    target_candidates = ["class", "target", "label", "type"]
    target = next(
        (column for column in target_candidates if column in data.columns),
        None
    )

    if target is None:
        raise ValueError(
            "Не найден целевой столбец грибов. "
            "Ожидается столбец class."
        )

    # Переносим целевой столбец в название class
    if target != "class":
        data = data.rename(columns={target: "class"})

    data["class"] = data["class"].astype(str).str.strip().str.lower()

    # Удаляем строки без класса
    data = data[data["class"].isin(["e", "p"])].copy()

    if data.empty:
        raise ValueError(
            "В столбце class должны быть значения e и p."
        )

    if data["class"].nunique() < 2:
        raise ValueError("В датасете должны присутствовать оба класса: e и p.")

    return data


def run_regression():
    """Линейная регрессия: прогноз итоговой оценки G3."""
    print("\n" + "=" * 55)
    print("ЗАДАНИЕ 1. ЛИНЕЙНАЯ РЕГРЕССИЯ")
    print("=" * 55)

    data = load_student_data()

    features = ["studytime", "failures", "g1", "g2"]
    target = "g3"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Количество записей: {len(data)}")
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Тестовая выборка: {len(X_test)}")

    print("\nМетрики качества:")
    print(f"MAE: {mae:.4f}")
    print(f"R²:  {r2:.4f}")

    print("\nКоэффициенты модели:")
    for name, coefficient in zip(features, model.coef_):
        print(f"{name}: {coefficient:.4f}")

    print(f"Свободный коэффициент: {model.intercept_:.4f}")

    # График G3 от G2 с линией регрессии
    plt.figure(figsize=(9, 6))

    plt.scatter(
        data["g2"],
        data["g3"],
        alpha=0.55,
        label="Фактические оценки"
    )

    # Для линии меняем G2, остальные признаки фиксируем
    x_line = np.linspace(data["g2"].min(), data["g2"].max(), 100)

    line_data = pd.DataFrame({
        "studytime": [data["studytime"].median()] * 100,
        "failures": [data["failures"].median()] * 100,
        "g1": [data["g1"].median()] * 100,
        "g2": x_line
    })

    y_line = model.predict(line_data)

    plt.plot(
        x_line,
        y_line,
        linewidth=2,
        label="Линия регрессии модели"
    )

    plt.xlabel("G2 — оценка за второй период")
    plt.ylabel("G3 — итоговая оценка")
    plt.title("Зависимость итоговой оценки G3 от G2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_classification():
    """Логистическая регрессия: определение съедобности грибов."""
    print("\n" + "=" * 55)
    print("ЗАДАНИЕ 2. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
    print("=" * 55)

    data = load_mushroom_data()

    X = data.drop(columns=["class"])
    y = data["class"]

    # Разделяем признаки на числовые и категориальные
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # One-Hot Encoding для категориальных признаков
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    transformers = []

    if categorical_features:
        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_features
        ))

    if numeric_features:
        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_features
        ))

    preprocessing = ColumnTransformer(transformers=transformers)

    model = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label="p", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label="p", zero_division=0
    )

    print(f"Количество записей: {len(data)}")
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Тестовая выборка: {len(X_test)}")

    print("\nМетрики качества:")
    print(f"Accuracy (точность классификации): {accuracy:.4f}")
    print(f"Precision (точность положительного класса): {precision:.4f}")
    print(f"Recall (полнота положительного класса): {recall:.4f}")

    print("\nКлассы:")
    print("e — съедобный гриб")
    print("p — ядовитый гриб")

    # Матрица ошибок
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["e", "p"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Съедобный (e)", "Ядовитый (p)"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Матрица ошибок — классификация грибов")
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 55)
    print("ЛАБОРАТОРНАЯ РАБОТА №2 — ВАРИАНТ 5")
    print("Линейные модели для регрессии и классификации")
    print("=" * 55)

    print("\nВыберите задание:")
    print("1 — Линейная регрессия (успеваемость студентов)")
    print("2 — Логистическая регрессия (классификация грибов)")
    print("0 — Выполнить оба задания")

    choice = input("\nВаш выбор: ").strip()

    try:
        if choice == "1":
            run_regression()
        elif choice == "2":
            run_classification()
        elif choice == "0":
            run_regression()
            run_classification()
        else:
            print("Ошибка: выберите 0, 1 или 2.")

    except FileNotFoundError as error:
        print(f"\nОШИБКА: {error}")
    except (ValueError, KeyError) as error:
        print(f"\nОШИБКА В ДАННЫХ: {error}")
    except Exception as error:
        print(f"\nНЕПРЕДВИДЕННАЯ ОШИБКА: {error}")

    input("\nНажмите Enter для выхода...")


if __name__ == "__main__":
    main()
