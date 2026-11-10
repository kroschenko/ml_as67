import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1500)

file_path = Path(__file__).resolve().parent / "Titanic-Dataset.csv"
df = pd.read_csv(file_path)

print("1. Первые 5 строк:")
print(df.head().to_string(index=False))

print("\nИнформация о столбцах:")
df.info()

print("\nКоличество пропусков:")
print(df.isnull().sum())

numeric_columns = ["Age", "SibSp", "Parch", "Fare"]

statistics = df[numeric_columns].agg(["mean", "median", "std"])
statistics.index = ["Среднее", "Медиана", "Стандартное отклонение"]

print("\nСтатистические показатели:")
print(statistics.round(2))

survival_counts = df["Survived"].value_counts().reindex([0, 1], fill_value=0)

print("\n2. Количество погибших и выживших:")
print(f"Погибли: {survival_counts[0]}")
print(f"Выжили: {survival_counts[1]}")

plt.figure(figsize=(7, 5))
bars = plt.bar(
    ["Погибли", "Выжили"],
    survival_counts.values,
    color=["tomato", "mediumseagreen"]
)
plt.bar_label(bars)
plt.title("Количество погибших и выживших пассажиров")
plt.ylabel("Количество пассажиров")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median)

print("\n3. Обработка пропусков в Age:")
print(f"Медианный возраст: {age_median}")
print(f"Осталось пропусков: {df['Age'].isnull().sum()}")

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df_encoded = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    dtype=int
)

encoded_columns = [
    column for column in df_encoded.columns
    if column.startswith(("Sex_", "Embarked_"))
]

print("\n4. Результат One-Hot Encoding:")
print(df_encoded[encoded_columns].head().to_string(index=False))

plt.figure(figsize=(8, 5))
plt.hist(df["Age"], bins=20, color="skyblue", edgecolor="black")
plt.title("Распределение возрастов пассажиров")
plt.xlabel("Возраст, лет")
plt.ylabel("Количество пассажиров")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

df["FamilySize"] = df["SibSp"] + df["Parch"]
df_encoded["FamilySize"] = df["FamilySize"]

print("\n5. Новый признак FamilySize:")
print(df[["SibSp", "Parch", "FamilySize"]].head(10).to_string(index=False))

normalization_columns = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
df_normalized = df_encoded.copy()

for column in normalization_columns:
    minimum = df_normalized[column].min()
    maximum = df_normalized[column].max()

    if maximum != minimum:
        df_normalized[column] = (
            (df_normalized[column] - minimum) / (maximum - minimum)
        )
    else:
        df_normalized[column] = 0.0

print("\nНормализация числовых признаков:")
print(df_normalized[normalization_columns].head().round(3).to_string(index=False))

print("\nДоля выживших в зависимости от пола, %:")
print((df.groupby("Sex")["Survived"].mean() * 100).round(2))

print("\nДоля выживших в зависимости от класса, %:")
print((df.groupby("Pclass")["Survived"].mean() * 100).round(2))

plt.show()