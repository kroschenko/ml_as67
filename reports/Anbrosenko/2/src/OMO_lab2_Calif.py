import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# План такой:
# 1) разбираемся с входными данными
# 2) Создаём модель
# 3) обучаем модель и в самом конце делаем предсказание для тестовой выборки
# 4) визуализируем
# 5) profit

# И так, первым делом, нужно взять данные, и заполнить пропуски, перед разбиением
# Там, совсем незаметно так, парочку значений в столбце total_bedrooms отсутствует
# Сначала настройки:
# Почему не 42? Пусть будет 43
SEED = 43
matplotlib.use('TkAgg')
# Добавим крутую сеточку в графики
sns.set_style("whitegrid")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # скрываю INFO и WARNING
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # явно запрещаем использование GPU, бо блин, спамит предупреждениями
import tensorflow as tf
from tensorflow import keras
# Очень много настроек, понимаю, глаза разбегаются


# ----- Пункт 1 -----
# Теперь к таблице. Воспользуемся навыками с первой лабы и заполним медианным значением (хотя и не просят):
df = pd.read_csv("california_housing.csv")
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
# Перемешиваем строки и сбрасываем индекс. Фрак=1 означает перемешивание,
# а ресет_индекс убирает старые индексы как отдельный столбец
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
# И переводим последний столбец в 0ли и 1цы с помощью One-Hot-Encoding
df_h = pd.get_dummies(df, columns=["ocean_proximity"])

# Теперь разбиваем данные на три выборки (на обучающую, валидационую и тестовую, первую и последнюю по заданию просят)
# Возьму 80%, 15% и 5% соответственно
n = len(df)
n_train = int(n * 0.80)
n_val = int(n * 0.15)

# Беру, делю на признаки и целевое значение. Привожу в массив нумпай, что и ожидает увидеть тензорфлоу
x_all = df_h.drop("median_house_value", axis=1).values.astype(np.float32)  # выходит np.ndarray
y_all = df_h["median_house_value"].values.astype(np.float32)

# Так, после того, как я разобрался что тут происходит, дополняю код:
x_train = x_all[:n_train]
y_train = y_all[:n_train]

# Расчитываю среднее
y_mean = y_train.mean()
# И отклонение
y_std = y_train.std()

x_val = x_all[n_train:n_train + n_val]
y_val = y_all[n_train:n_train + n_val]

x_test = x_all[n_train + n_val:]
y_test = y_all[n_train + n_val:]
# С подготовкой вроде закончено, переходим к, собственно, нейросети (PS: нет)

# Теперь делаем самое важное:
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std


# ----- Пункт 2 -----
# Нормализуем данные, чтобы были от 0 до 1
# Тут следовало бы убрать данные, которые принимают 0 или 1, но я не уверен как. Но это некритично
normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(x_train)   # статистики только по обучающей выборке

# Создаём модель с одним нейроном в скрытом слое
model = keras.Sequential([
    normalizer,
    keras.layers.Dense(1)   # Пометка. y = w·x + b
])

# Оптимизатор ставим Адам, так как знаю как работает только он и ищем нужные данные
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    # Метод наименьших квадратов
    loss="mse",
    # Поиск R2 (Coefficient of Determination). Честно, всё ещё не уверен, что это и зачем
    metrics=[keras.metrics.R2Score()]
)

# ----- Пункт 3 -----
# Обучаем нашу модель
# Ну тут вродь всё понятно, разве что verbose=0 отключает вывод прогресса обучения модели
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=15,
    batch_size=32,
    verbose=0
)
# Блиин, сколько геморроя было-то! Я 3 часа учил и пытался понять, что за чёрт!
# Но такая огромная ошибка взялась из-за того, что выходное значение большое. Я думал я с ума сошёл
# Короче, всё фигня, давай по новой

# Печатает нашу модель
model.summary()

# ----- Пункт 4 ----
# Визуализация, самое сложное позади
income_test = df["median_income"].values[n_train + n_val:]

# Линия простой линейной регрессии y = k·x + b по одному признаку
k, b = np.polyfit(income_test, y_test, deg=1)
r = np.corrcoef(income_test, y_test)[0, 1]

# Только одна странность, почему цены ограничиваются 500000, ну да ладно
x_line = np.linspace(income_test.min(), income_test.max(), 100)
y_line = k * x_line + b

# Finally!!
plt.plot(history.history['loss'])
plt.show()

plt.figure(figsize=(10, 6))
# Теперь умножаем выходное значение для правильности
plt.scatter(income_test, y_test * y_std + y_mean, alpha=0.4, s=15, color="steelblue",
            label="Тестовая выборка (факт)")
plt.plot(x_line, y_line * y_std + y_mean, color="pink", linewidth=2,
         label=f"Линия регрессии: y = {k * y_std + y_mean:.0f}·x + {b * y_std + y_mean:.0f}")
plt.xlabel("Медианный доход (median_income)")
plt.ylabel("Медианная стоимость дома (median_house_value)")
plt.title(f"Диаграмма рассеяния, коэффициент корреляции r = {r:.3f}")
plt.legend()
plt.tight_layout()
plt.show()

# ----- Пункт 5 -----
# (=^･ω･^=) Профит
