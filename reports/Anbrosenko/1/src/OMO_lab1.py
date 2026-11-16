import matplotlib
# Чтобы не жаловалось:
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Добавим крутую сеточку в графики
sns.set_style("whitegrid")
# И увеличим выход для таблиц
pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', 10000)

# Далее по заданию
# 1. Загрузка данных и просмотр
df = pd.read_csv("Titanic-Dataset.csv")

print("Первые 5 строк:")
print(df.head())
print("\nИнформация о столбцах:")
print(df.info())

# 2. Количество выживших и погибших — столбчатая диаграмма
survived_counts = df["Survived"].value_counts()

plt.figure(figsize=(5, 4))
# Мне красный глаза режет, поэтому организована милота
plt.bar(["Погиб (0)", "Выжил (1)"], survived_counts.values, color=["pink", "green"])
plt.title("Количество выживших и погибших")
plt.ylabel("Число пассажиров")
plt.tight_layout()
plt.show()

# 3. Заполнение пропусков в Age медианным значением и взятие данных для гистограммы
age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median)
print(f"\nПропуски в Age заполнены медианой: {age_median}")

# 4. One-Hot Encoding для Sex и Embarked
df = pd.get_dummies(df, columns=["Sex", "Embarked"], prefix=["Sex", "Embarked"])
print("\nПосле преобразования (список столбцов):")
print(df.columns.tolist())

# 5. Гистограмма распределения возрастов
plt.figure(figsize=(6, 4))
# По началу я выводил "Неизвестно" как отдельный столбец, но мы же специально добавили медианное значение =/
plt.hist(df["Age"], edgecolor="black")
plt.title("Распределение возрастов пассажиров")
plt.xlabel("Возраст")
plt.ylabel("Частота")
plt.tight_layout()
plt.show()

# 6. Новый признак FamilySize = SibSp + Parch
df["FamilySize"] = df["SibSp"] + df["Parch"]
print("\nНовый признак FamilySize создан.")
print(df[["SibSp", "Parch", "FamilySize"]].head(10))
# ฅ^•ﻌ•^ฅ    - Лапки вверх!
