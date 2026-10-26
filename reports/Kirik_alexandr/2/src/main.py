import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def regression_solution() -> None:
    FEATURES : list[str] = ['Length1', 'Length2', 'Length3', 'Height', 'Width']

    TARGET_FEATURE : str = 'Weight'
    fish_dataset : pd.DataFrame = pd.read_csv('Fish.csv', sep=',')

    # Обучение модели линейной регрессии
    X_train, X_test, y_train, y_test = train_test_split(
        fish_dataset[FEATURES],
        fish_dataset[TARGET_FEATURE],
        test_size=0.2,
        random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Оценка модели
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"R^2: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    
    # Визуализация данных
    fig, ax = plt.subplots()
    plt.scatter(x=fish_dataset[FEATURES[2]],
                y=fish_dataset[TARGET_FEATURE],
                alpha=0.4,
                c='green'
                )
    # Добавление линии регрессии
    z : np.ndarray = np.polyfit(
                fish_dataset[FEATURES[2]], 
                fish_dataset[TARGET_FEATURE], 
                1
                )
    p = np.poly1d(z)
    plt.plot(fish_dataset[FEATURES[2]], 
             p(fish_dataset[FEATURES[2]]), 
             color='red', 
             label='Линия регрессии')
    plt.title(f"Диаграмма рассеивания '{FEATURES[2]}' и '{TARGET_FEATURE}' с линией регрессии")
    plt.legend()
    plt.show()

def classification_solution() -> None:
    TARGET : str = 'y'
    bank_dataset : pd.DataFrame = pd.read_csv('bank.csv', sep=';')
    # print(bank_dataset.groupby('poutcome')['poutcome'].count())
    # Преобразование столбцов 
    one_hot_encoding_col : list[str] = ['job', 'education', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome']
    for i in one_hot_encoding_col:
        bank_dataset = pd.get_dummies(
            bank_dataset,
            columns=[i],
            prefix=i,
            drop_first=True
        )

    # Преобразование месяцев в числовые значение по порядку месяцев
    bank_dataset['month'] = bank_dataset['month'].astype('category')
    # months : list[str] = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    bank_dataset['month'] = bank_dataset['month'].cat.codes

    # Обучение модели логистической регрессии
    features : list[str] = bank_dataset.columns.to_list()
    # print(features)
    features.remove(TARGET)
    # print("Размерность y: ", bank_dataset[TARGET].shape)
    X_train, X_test, y_train, y_test = train_test_split(
            bank_dataset[features],
            bank_dataset[TARGET].values.ravel(),
            test_size=0.25,
            random_state=42
    )


    model = LogisticRegression()

    # Нормализация тренировочных и тестовых данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)

    # Оценка модели
    acc = accuracy_score(y_test, y_pred)
    ps = precision_score(y_test, y_pred, pos_label="yes")
    rs = recall_score(y_test, y_pred, pos_label="yes")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {ps:.3f}")
    print(f"Recall: {rs:.3f}")

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["no", "yes"])
    disp.plot()
    plt.title("Матрица ошибок")
    plt.show()




if __name__ == "__main__":
    print("ОБУЧЕНИЕ МОДЕЛИ ЛИНЕЙНОЙ РЕГРЕССИИ:")
    regression_solution()
    print("ОБУЧЕНИЕ МОДЕЛИ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ И РАСЧЕТ МАТРИЦЫ ОШИБОК:")
    classification_solution()