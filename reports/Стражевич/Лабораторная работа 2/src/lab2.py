
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

#1. ЗАГРУЗКА ДАННЫХ
print("ЗАГРУЗКА ДАННЫХ")

#загружаем датасет красного вина
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

data = pd.read_csv(url, sep=";")

print("\nПервые пять строк датасета:")    #первые пять строк
print(data.head())

print("\nРазмер датасета:")    #количество строк и столбцов
print(data.shape)

print("\nКоличество пропусков:")
print(data.isnull().sum())

#ЛИНЕЙНАЯ РЕГРЕССИЯ
print("\n\nЛИНЕЙНАЯ РЕГРЕССИЯ")

#выбираем все признаки, кроме quality (целевая переменная)
X = data.drop("quality", axis=1)

y = data["quality"]    #целевая переменная

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

regression_model = LinearRegression()    #модель линейной регрессии

regression_model.fit(X_train, y_train)

y_pred = regression_model.predict(X_test)    #прогнозы модели

mse = mean_squared_error(y_test, y_pred)    #метрики качества
r2 = r2_score(y_test, y_pred)

print("\nРезультаты линейной регрессии:")
print("MSE:", round(mse, 4))
print("R2:", round(r2, 4))

#3. ГРАФИК ЗАВИСИМОСТИ QUALITY ОТ ALCOHOL
#выбираем признак alcohol
alcohol = data[["alcohol"]]
quality = data["quality"]

plot_model = LinearRegression()

plot_model.fit(alcohol, quality)    #модель на признаке alcohol

quality_line = plot_model.predict(alcohol)
sort_index = np.argsort(alcohol["alcohol"].values)

plt.figure(figsize=(9, 6))    #график

#значения качества вина
plt.scatter(
    alcohol["alcohol"],
    quality,
    alpha=0.4,
    label="Реальные значения"
)

#линию регрессии
plt.plot(
    alcohol["alcohol"].values[sort_index],
    quality_line[sort_index],
    color="red",
    linewidth=2,
    label="Линия регрессии"
)

plt.title("Зависимость качества вина от содержания алкоголя")
plt.xlabel("Содержание алкоголя (alcohol)")
plt.ylabel("Качество вина (quality)")
plt.legend()
plt.grid(True)

plt.savefig("regression_graph.png", dpi=300, bbox_inches="tight")
plt.show()


#4. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
print("\n\nЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")

data["good_wine"] = (data["quality"] >= 7).astype(int)

X_class = data.drop(["quality", "good_wine"], axis=1)
y_class = data["good_wine"]    #целевая переменная классификации

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

#модель логистической регрессии
classification_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

classification_model.fit(X_train_class, y_train_class)    #обучаем модель

y_pred_class = classification_model.predict(X_test_class)


#5. РАСЧЁТ МЕТРИК КЛАССИФИКАЦИИ
accuracy = accuracy_score(y_test_class, y_pred_class)

#Precision для класса "хорошее вино" 
precision = precision_score(
    y_test_class,
    y_pred_class,
    pos_label=1,
    zero_division=0
)

#Recall для класса "хорошее вино" 
recall = recall_score(
    y_test_class,
    y_pred_class,
    pos_label=1,
    zero_division=0
)

print("\nРезультаты логистической регрессии:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))


#6. МАТРИЦА ОШИБОК

cm = confusion_matrix(
    y_test_class,
    y_pred_class,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Плохое вино (0)", "Хорошее вино (1)"],
    yticklabels=["Плохое вино (0)", "Хорошее вино (1)"]
)

plt.title("Матрица ошибок")
plt.xlabel("Предсказанный класс")
plt.ylabel("Настоящий класс")

plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()



print("\n\nРАБОТА ЗАВЕРШЕНА!")
print("Графики сохранены:")
print("1. regression_graph.png")
print("2. confusion_matrix.png")