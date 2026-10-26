import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.figsize"] = (10, 6)
df = pd.read_csv("iris.csv")

print("1. Первые 5 строк датасета")
print(df.head())
print("\n2. Информация о типах столбцов (.info())")
df.info()
print("\n3. Основные статистические показатели (.describe())")
print(df.describe().T[["mean", "50%", "std"]])
print("\n4. Проверка на пропущенные значения")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:
    df = df.fillna(df.median(numeric_only=True))
    print("Пропуски обработаны.")
else:
    print("Пропуски в данных отсутствуют.")

# Количество образцов каждого вида
print("\nРаспределение по видам ирисов")
print(df["variety"].value_counts())
numeric_features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

# Парные диаграммы рассеяния  с легендой
species_colors = {"Setosa": "red", "Versicolor": "green", "Virginica": "blue"}
colors = df["variety"].map(species_colors)

axes = pd.plotting.scatter_matrix(
    df[numeric_features],
    c=colors,
    figsize=(10, 10),
    diagonal="hist",
    alpha=0.8,
    s=40,
)

# Создаем элемент легенды
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=color,
        markersize=10,
        label=species,
    )
    for species, color in species_colors.items()
]

# Добавляем легенду в верхний правый угол фигуры
plt.figlegend(
    handles=handles,
    title="Вид ириса",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    fontsize=10,
    title_fontsize=11,
)

plt.suptitle("Парные диаграммы рассеяния для признаков Ириса", y=0.98, fontsize=14)
plt.subplots_adjust(top=0.93)
plt.show()

# Средние значения по каждому виду
print("\nСредние значения признаков по видам ирисов")
print(df.groupby("variety")[numeric_features].mean())

# Ящик с усами 
fig, ax = plt.subplots(figsize=(8, 6))
df.boxplot(column="petal.length", by="variety", grid=True, ax=ax)
plt.title("Распределение Petal Length по видам ирисов")
plt.suptitle("")
plt.xlabel("Вид ириса")
plt.ylabel("Petal Length (см)")
plt.tight_layout()
plt.show()
# One-Hot Encoding и стандартизация
df_encoded = pd.get_dummies(df, columns=["variety"], drop_first=False, dtype=int)

df_scaled = df_encoded.copy()
df_scaled[numeric_features] = (
    df[numeric_features] - df[numeric_features].mean()
) / df[numeric_features].std()

print("\nСтатистика после ручной стандартизации (mean ≈ 0, std = 1)")
print(df_scaled[numeric_features].describe().T[["mean", "std"]])