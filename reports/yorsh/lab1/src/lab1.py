import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "winequality-white.csv", sep=";")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1500)

print("Первые 5 строк:")
print(df.head().to_string(index=False))

print("\n1. Информация о типах столбцов:")
df.info()

print("\nКоличество пропусков:")
print(df.isnull().sum())

numeric_columns = df.select_dtypes(include=np.number).columns

print("\nОсновные статистические показатели:")
statistics = df[numeric_columns].agg(["mean", "median", "std"])
statistics.index = ["Среднее", "Медиана", "Стандартное отклонение"]
print(statistics.round(3).to_string())

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

df["QualityCategory"] = np.select(
    [df["quality"] <= 4, df["quality"] <= 6],
    ["плохое", "среднее"],
    default="хорошее"
)

print("\n2. Категории качества:")
print(df[["quality", "QualityCategory"]].head(10).to_string(index=False))

category_counts = df["QualityCategory"].value_counts().reindex(
    ["плохое", "среднее", "хорошее"],
    fill_value=0
)

print("\n3. Количество вин каждой категории:")
print(category_counts)

plt.figure(figsize=(8, 5))
bars = plt.bar(
    category_counts.index,
    category_counts.values,
    color=["tomato", "gold", "mediumseagreen"]
)
plt.bar_label(bars)
plt.title("Распределение вин по категориям качества")
plt.xlabel("Категория качества")
plt.ylabel("Количество вин")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

correlation = df["fixed acidity"].corr(df["pH"])

print("\n4. Корреляция между fixed acidity и pH:")
print(f"Коэффициент корреляции: {correlation:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(
    df["fixed acidity"],
    df["pH"],
    alpha=0.4,
    s=20,
    color="steelblue"
)
plt.title("Зависимость pH от фиксированной кислотности")
plt.xlabel("Фиксированная кислотность (fixed acidity)")
plt.ylabel("pH")
plt.grid(alpha=0.3)
plt.tight_layout()

q1 = df[numeric_columns].quantile(0.25)
q3 = df[numeric_columns].quantile(0.75)
iqr = q3 - q1

lower_bounds = q1 - 1.5 * iqr
upper_bounds = q3 + 1.5 * iqr

outlier_counts = (
    (df[numeric_columns] < lower_bounds)
    | (df[numeric_columns] > upper_bounds)
).sum().sort_values(ascending=False)

print("\n5. Количество выбросов по правилу 1.5 IQR:")
print(outlier_counts)

max_outliers = outlier_counts.max()
outlier_features = outlier_counts[
    outlier_counts == max_outliers
].index.tolist()

print("\nПризнаки с наибольшим количеством выбросов:")
for column in outlier_features:
    print(f"{column}: {max_outliers}")

df[numeric_columns].plot(
    kind="box",
    subplots=True,
    layout=(3, 4),
    figsize=(16, 11),
    sharey=False
)
plt.suptitle("Поиск выбросов в числовых признаках")
plt.tight_layout(rect=[0, 0, 1, 0.96])

df_standardized = df.copy()

means = df[numeric_columns].mean()
stds = df[numeric_columns].std(ddof=0)
stds = stds.replace(0, 1)

df_standardized[numeric_columns] = (
    df[numeric_columns] - means
) / stds

print("\n6. Стандартизированные данные — первые 5 строк:")
print(df_standardized.head().round(3).to_string(index=False))

print("\nПроверка стандартизации:")
check = pd.DataFrame({
    "Среднее": df_standardized[numeric_columns].mean(),
    "Стандартное отклонение": df_standardized[numeric_columns].std(ddof=0)
})
print(check.round(6).to_string())

plt.show()