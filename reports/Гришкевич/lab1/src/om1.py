
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / "BostonHousing.csv")

data.columns = data.columns.str.strip()

print("Количество записей:", len(data))
print("Количество признаков:", len(data.columns))

print("\nСТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ")
print(data.describe().round(3))

analysis_data = data.drop(columns=["CAT. MEDV"], errors="ignore")

correlation = analysis_data.corr(numeric_only=True)

plt.figure(figsize=(13, 10))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Матрица корреляции признаков Boston Housing")
plt.tight_layout()
plt.savefig(folder / "correlation_heatmap.png")
plt.show()

correlations = correlation["MEDV"].drop("MEDV")
best_feature = correlations.abs().idxmax()
best_value = correlations[best_feature]

print("\nКОРРЕЛЯЦИЯ С MEDV")
print(correlations.sort_values(key=abs, ascending=False).round(4))

print("\nНаиболее сильно коррелирующий признак:", best_feature)
print("Коэффициент корреляции:", round(best_value, 4))

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=data,
    x=best_feature,
    y="MEDV"
)
plt.title(f"Зависимость MEDV от {best_feature}")
plt.xlabel(best_feature)
plt.ylabel("MEDV")
plt.tight_layout()
plt.savefig(folder / "scatter_plot.png")
plt.show()

normalized_data = data.copy()

numeric_columns = normalized_data.select_dtypes(
    include=["number"]
).columns

scaler = MinMaxScaler()

normalized_data[numeric_columns] = scaler.fit_transform(
    normalized_data[numeric_columns]
)

normalized_data.to_csv(
    folder / "BostonHousing_normalized.csv",
    index=False
)

print("\nНОРМАЛИЗАЦИЯ")
print("Все числовые столбцы приведены к диапазону от 0 до 1.")
print("Минимальные значения:")
print(normalized_data[numeric_columns].min().round(3))

print("\nМаксимальные значения:")
print(normalized_data[numeric_columns].max().round(3))

plt.figure(figsize=(8, 6))
sns.histplot(
    data=data,
    x="CRIM",
    bins=30,
    kde=True
)
plt.title("Распределение уровня преступности (CRIM)")
plt.xlabel("CRIM")
plt.ylabel("Количество районов")
plt.tight_layout()
plt.savefig(folder / "crim_histogram.png")
plt.show()

print("\nРабота программы завершена.")
