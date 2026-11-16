
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. ЗАГРУЗКА ДАННЫХ
# ==========================================

df = pd.read_csv("adult.csv")

# Убираем лишние пробелы в текстовых столбцах
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

print("ПЕРВЫЕ 10 СТРОК:")
print(df.head(10))

print("\nОБЩАЯ ИНФОРМАЦИЯ О ДАННЫХ:")
df.info()

# ==========================================
# 2. ОБРАБОТКА ПРОПУСКОВ В WORKCLASS
# ==========================================

print("\nКоличество значений '?' в workclass до обработки:")
print((df["workclass"] == "?").sum())

# Заменяем '?' на наиболее часто встречающееся значение
mode_workclass = df.loc[
    df["workclass"] != "?", "workclass"
].mode()[0]

df["workclass"] = df["workclass"].replace("?", mode_workclass)

print("\nНаиболее часто встречающееся значение workclass:")
print(mode_workclass)

print("\nКоличество '?' после обработки:")
print((df["workclass"] == "?").sum())

# ==========================================
# 3. КОЛИЧЕСТВО МУЖЧИН И ЖЕНЩИН
# ==========================================

gender_counts = df["sex"].value_counts()

print("\nКОЛИЧЕСТВО МУЖЧИН И ЖЕНЩИН:")
print(gender_counts)

plt.figure(figsize=(8, 5))
gender_counts.plot(kind="bar", color=["steelblue", "salmon"])

plt.title("Количество мужчин и женщин")
plt.xlabel("Пол")
plt.ylabel("Количество человек")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# ==========================================
# 4. ПРЕОБРАЗОВАНИЕ RACE В ЧИСЛОВОЙ ФОРМАТ
# ==========================================

print("\nКатегории признака race:")
print(df["race"].unique())

# One-Hot Encoding: каждая категория становится отдельным столбцом
df = pd.get_dummies(
    df,
    columns=["race"],
    prefix="race",
    dtype=int
)

print("\nДанные после преобразования race:")
print(df.head())

# ==========================================
# 5. ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ ВОЗРАСТА
# ==========================================

plt.figure(figsize=(10, 6))

income_groups = ["<=50K", ">50K"]
colors = ["skyblue", "salmon"]

for income, color in zip(income_groups, colors):
    ages = df.loc[df["income"] == income, "age"]
    plt.hist(
        ages,
        bins=20,
        alpha=0.6,
        label=income,
        color=color
    )

plt.title("Распределение возраста по уровню дохода")
plt.xlabel("Возраст")
plt.ylabel("Количество человек")
plt.legend(title="Доход")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# ==========================================
# 6. СОЗДАНИЕ ПРИЗНАКА IS_USA
# ==========================================

# 1 — человек родился в США
# 0 — человек родился в другой стране

df["is_usa"] = np.where(
    df["native.country"] == "United-States",
    1,
    0
)

print("\nКоличество людей из США и других стран:")
print(df["is_usa"].value_counts())

# ==========================================
# ДОПОЛНИТЕЛЬНО: СОХРАНЕНИЕ ОБРАБОТАННЫХ ДАННЫХ
# ==========================================

df.to_csv("adult_processed.csv", index=False)

print("\nОбработанные данные сохранены в adult_processed.csv")

# ==========================================
# ОТОБРАЖЕНИЕ ГРАФИКОВ
# ==========================================

plt.show()
