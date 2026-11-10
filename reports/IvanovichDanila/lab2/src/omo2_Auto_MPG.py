import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Настройки для графиков
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    print("--- ЗАДАЧА 1: РЕГРЕССИЯ (Auto MPG) ---")

    # 1. Загрузка данных
    try:
        df_mpg = pd.read_csv('auto-mpg.csv', na_values='?')
    except FileNotFoundError:
        print("Ошибка: Файл 'auto-mpg.csv' не найден в текущей директории.")
        return

    # 2. Обработка пропусков (заполняем медианой)
    df_mpg['horsepower'] = df_mpg['horsepower'].fillna(df_mpg['horsepower'].median())

    # Выбор признаков по заданию
    features = ['cylinders', 'horsepower', 'weight']
    target = 'mpg'

    X = df_mpg[features]
    y = df_mpg[target]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Обучение модели линейной регрессии
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Предсказание
    y_pred = lin_reg.predict(X_test)

    # 4. Расчет метрик
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Метрика MSE: {mse:.2f}")
    print(f"Метрика R^2: {r2:.4f}")
    print(f"Коэффициенты модели: {dict(zip(features, lin_reg.coef_.round(2)))}")
    print(f"Свободный член (intercept): {lin_reg.intercept_:.2f}")

    # 5. Визуализация
    plt.figure(figsize=(8, 5))
    sns.regplot(x='horsepower', y='mpg', data=df_mpg, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title('Зависимость расхода топлива (mpg) от мощности (horsepower)')
    plt.xlabel('Мощность (horsepower)')
    plt.ylabel('Расход топлива (mpg)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()