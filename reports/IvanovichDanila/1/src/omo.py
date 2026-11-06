import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных и проверка пропущенных значений
df = pd.read_csv("iris.csv")

print("Первые 5 строк данных:")
print(df.head())

print("\nРазмер набора данных:")
print(df.shape)

print("\nКоличество пропущенных значений:")
print(df.isnull().sum())

if df.isnull().sum().sum() == 0:
    print("Пропущенных значений нет.")
else:
    print("В данных есть пропущенные значения.")

# Названия признаков и столбец с видами ирисов
features = [
    "sepal.length",
    "sepal.width",
    "petal.length",
    "petal.width"
]
target = "variety"

# 2. Количество образцов каждого вида ириса
print("\nКоличество образцов каждого вида ириса:")
print(df[target].value_counts())

# 3. Парные диаграммы рассеяния (Pair Plot)
sns.pairplot(
    df,
    vars=features,
    hue=target,
    diag_kind="hist"
)

plt.suptitle("Парные диаграммы рассеяния для Iris", y=1.02)
plt.show()

# 4. Среднее значение каждого признака для каждого вида ириса
means = df.groupby(target)[features].mean()

print("\nСредние значения признаков для каждого вида ириса:")
print(means.round(2))

# 5. Ящик с усами для Petal Length
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x=target,
    y="petal.length"
)

plt.title("Распределение длины лепестка по видам ирисов")
plt.xlabel("Вид ириса")
plt.ylabel("Petal Length (cm)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 6. Стандартизация признаков
scaler = StandardScaler()

df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

print("\nПервые 5 строк стандартизированных данных:")
print(df_scaled.head().round(3))

print("\nСредние значения после стандартизации:")
print(df_scaled[features].mean().round(3))

print("\nСтандартные отклонения после стандартизации:")
print(df_scaled[features].std(ddof=0).round(3))

# Сохранение стандартизированных данных в отдельный файл
df_scaled.to_csv("iris_standardized.csv", index=False)

print("\nСтандартизированные данные сохранены в iris_standardized.csv")