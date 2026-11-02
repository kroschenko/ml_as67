import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

# 1. Регрессия


def regression_task():
    try:
        df = pd.read_csv('winequality-white.csv', sep=';')
    except FileNotFoundError:
        print("Ошибка: Файл 'winequality-white.csv' не найден. "
              "Скачайте датасет и укажите правильный путь.")
        return

    target = 'quality'
    if target not in df.columns:
        print("Ошибка: колонка 'quality' не найдена.")
        return

    df_reg = df.dropna().copy()
    X = df_reg.drop(columns=[target])
    y = df_reg[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Регрессия (предсказание quality)")
    print(f"MSE: {mse:.4f}")
    print(f"R2  (коэффициент детерминации): {r2:.4f}")

    # Визуализация: quality vs alcohol + линия регрессии
    plt.figure(figsize=(10, 6))
    plt.scatter(df_reg['alcohol'], df_reg['quality'],
                alpha=0.4, color='blue', label='Данные')

    z = np.polyfit(df_reg['alcohol'], df_reg['quality'], 1)
    p = np.poly1d(z)
    xs = np.linspace(df_reg['alcohol'].min(), df_reg['alcohol'].max(), 100)
    plt.plot(xs, p(xs), color='red', linewidth=2, label='Линия регрессии')

    plt.title('Зависимость качества вина от содержания алкоголя')
    plt.xlabel('Alcohol')
    plt.ylabel('Quality')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# 2. Классификация(7>= - хорошо)


def classification_task():
    try:
        df = pd.read_csv('winequality-white.csv', sep=';')
    except FileNotFoundError:
        print("Ошибка: Файл 'winequality-white.csv' не найден. "
              "Скачайте датасет и укажите правильный путь.")
        return

    if 'quality' not in df.columns:
        print("Ошибка: колонка 'quality' не найдена.")
        return

    df = df.dropna().copy()

    # Новый бинарный целевой столбец
    df['target'] = (df['quality'] >= 7).astype(int)

    X = df.drop(columns=['quality', 'target'])
    y = df['target']

    print("\nКлассификация (good/bad)")
    print("Распределение классов:")
    print(y.value_counts().rename({0: 'bad (<7)', 1: 'good (>=7)'}))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight='balanced',
                               random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision (good): {precision:.4f}")
    print(f"Recall    (good): {recall:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['bad (<7)', 'good (>=7)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Матрица ошибок (Wine Quality)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    regression_task()
    classification_task()