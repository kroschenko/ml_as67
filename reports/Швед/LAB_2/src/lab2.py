import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42


def regression_task():
    print("=" * 60)
    print("РЕГРЕССИЯ: стоимость жилья в Калифорнии (California Housing)")
    print("=" * 60)

    try:
        df = pd.read_csv(os.path.join(BASE_DIR, 'california_housing.csv'))
    except FileNotFoundError:
        print("Ошибка: файл 'california_housing.csv' не найден.")
        return

    print(f"Размер датасета: {df.shape}")
    print(f"Пропуски:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Заполняем пропуски в total_bedrooms медианой
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    # Категориальный признак ocean_proximity -> one-hot кодирование
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True, dtype=int)

    target = 'median_house_value'
    X = df.drop(columns=[target])
    y = df[target]

    # Разделение на обучающую (80%) и тестовую (20%) выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Обучающая выборка: {X_train.shape[0]}, тестовая: {X_test.shape[0]}")

    # Обучение модели линейной регрессии на всех признаках
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказания для тестовой выборки
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nМодель на всех признаках:")
    print(f"MSE  (среднеквадратичная ошибка): {mse:,.2f}")
    print(f"RMSE (корень из MSE):             {np.sqrt(mse):,.2f}")
    print(f"R2   (коэффициент детерминации):  {r2:.4f}")

    print("\nКоэффициенты модели:")
    coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
    for name, value in coefs.items():
        print(f"  {name:28s} {value:14,.2f}")

    # Парная регрессия median_house_value ~ median_income (для линии на графике)
    model_income = LinearRegression()
    model_income.fit(X_train[['median_income']], y_train)
    y_pred_income = model_income.predict(X_test[['median_income']])
    mse_income = mean_squared_error(y_test, y_pred_income)
    r2_income = r2_score(y_test, y_pred_income)
    print("\nМодель только на median_income:")
    print(f"y = {model_income.coef_[0]:,.2f} * median_income + {model_income.intercept_:,.2f}")
    print(f"MSE: {mse_income:,.2f}, RMSE: {np.sqrt(mse_income):,.2f}, R2: {r2_income:.4f}")

    # Диаграмма рассеяния median_income vs median_house_value с линией регрессии
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['median_income'], y_test, alpha=0.3, s=10,
                color='steelblue', label='Тестовые данные')
    x_line = np.linspace(X['median_income'].min(), X['median_income'].max(), 100)
    y_line = model_income.predict(pd.DataFrame({'median_income': x_line}))
    plt.plot(x_line, y_line, color='red', linewidth=2,
             label=f'Линия регрессии (R² = {r2_income:.3f})')
    plt.title('Зависимость стоимости жилья от медианного дохода')
    plt.xlabel('median_income (медианный доход, десятки тыс. $)')
    plt.ylabel('median_house_value (медианная стоимость дома, $)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'regression_plot.png'), dpi=150)
    plt.show()

    # Дополнительно: предсказанные значения против фактических (модель на всех признаках)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color='seagreen')
    lims = [y_test.min(), y_test.max()]
    plt.plot(lims, lims, 'r--', linewidth=2, label='Идеальное предсказание')
    plt.title(f'Предсказание vs факт (все признаки, R² = {r2:.3f})')
    plt.xlabel('Фактическая стоимость, $')
    plt.ylabel('Предсказанная стоимость, $')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'regression_pred_vs_true.png'), dpi=150)
    plt.show()


def classification_task():
    print("\n" + "=" * 60)
    print("КЛАССИФИКАЦИЯ: выживаемость на Титанике (Titanic)")
    print("=" * 60)

    try:
        df = pd.read_csv(os.path.join(BASE_DIR, 'titanic.csv'))
    except FileNotFoundError:
        print("Ошибка: файл 'titanic.csv' не найден.")
        return

    print(f"Размер датасета: {df.shape}")
    print(f"Пропуски до обработки:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Удаляем признаки, не несущие полезной информации для модели
    # (Cabin пропущен у 77% пассажиров)
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # Заполнение пропусков
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    print(f"Пропусков после обработки: {df.isnull().sum().sum()}")

    # Преобразование категорий в числа
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True, dtype=int)

    X = df.drop(columns=['Survived'])
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"Обучающая выборка: {X_train.shape[0]}, тестовая: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение логистической регрессии
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print("\nКоэффициенты модели (на стандартизированных признаках):")
    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    for name, value in coefs.items():
        print(f"  {name:12s} {value:8.4f}")

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\nМатрица ошибок:")
    print(cm)
    print(f"TN (верно: не выжил) = {tn}")
    print(f"FP (ошибочно: выжил) = {fp}")
    print(f"FN (ошибочно: не выжил) = {fn}")
    print(f"TP (верно: выжил) = {tp}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Не выжил (0)', 'Выжил (1)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'), dpi=150)
    plt.show()


if __name__ == "__main__":
    regression_task()
    classification_task()
