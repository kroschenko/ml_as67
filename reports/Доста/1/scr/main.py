import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('BostonHousing.csv')

print("\nПервые 5 строк:")
print(df.head())


#1 основные статистические характеристики (.describe())
print("\nОписательные статистики (.describe()):")
print(df.describe())


#2 матрица корреляции с помощью тепловая карта (heatmap)
corr = df.corr()

plt.figure(figsize=(14, 10))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar(label="Коэффициент корреляции")

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(
            j, i, f"{corr.iloc[i, j]:.2f}",
            ha="center", va="center",
            color="black", fontsize=8
        )

plt.title("Корреляционная матрица Boston Housing")
plt.tight_layout()
plt.show()


#3 признак наиболее сильно коррелирующий с MEDV (медианная стоимость дома)
corr_medv = corr["MEDV"].drop("MEDV").sort_values(key=abs, ascending=False)

top_feature = corr_medv.index[1]
top_value = corr_medv.iloc[1]

print(f"\nНаиболее сильная корреляция с MEDV: {top_feature}")
print(f"Коэффициент корреляции: {top_value:.3f}")


#4 Диаграмма рассеяния для top_feature и MEDV
plt.figure(figsize=(8, 6))
plt.scatter(df[top_feature], df["MEDV"], alpha=0.7)
plt.xlabel(top_feature)
plt.ylabel("MEDV (медианная стоимость дома)")
plt.title(f"Зависимость MEDV от {top_feature}")
plt.grid(True)
plt.show()


#5 Нормализация числовых признаков к диапазону [0, 1]
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

X_norm = (X - X.min()) / (X.max() - X.min())

df_norm = X_norm.copy()
df_norm["MEDV"] = y.values


#6 Гистограмма распределения CRIM
plt.figure(figsize=(10, 6))
plt.hist(df["CRIM"], bins=40, color="skyblue", edgecolor="black")
plt.xlabel("CRIM (уровень преступности)")
plt.ylabel("Частота")
plt.title("Распределение уровня преступности CRIM")
plt.grid(axis="y", alpha=0.3)
plt.show()