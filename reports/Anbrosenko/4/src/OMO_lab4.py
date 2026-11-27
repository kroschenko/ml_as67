import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Честно, я смирился с ошибками, пусть будут
from tensorflow import keras

# Настройки
SEED = 46
matplotlib.use('TkAgg')
# Добавим крутую сеточку в графики
sns.set_style("whitegrid")
features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']



# ----- Пункт 1 -----
# Берём наш датасет
df = pd.read_csv("iris.csv")


# Далее, берём и перемешиваем инфу.
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# Теперь разбиваем данные две выборки, тестовую и тренировочную
n = len(df)
n_train = int(n * 0.90)

x_all = df[features].values.astype(np.float32)
df = pd.get_dummies(df, columns=["variety"])
df.drop(columns=features, inplace=True)
y_all = df

x_train = x_all[:n_train]
y_train = y_all[:n_train]

x_test = x_all[n_train:]
y_test = y_all[n_train:]
# С подготовкой вроде закончено, переходим к, собственно, нейросети


# ----- Этап 2 -----
# Всё то же самое, нормализуем данные, делаем модель и переходим к 3ему этапу
normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(x_train)

model = keras.Sequential([
    normalizer,
    keras.layers.Dense(5),
    keras.layers.Dense(3, activation="softmax")
])

# Правила обучения
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    # Категориальная кросс энтропия, лучший выбор для категорирования между более двумя вариантами, как в данном случае
    loss="categorical_crossentropy",
    metrics=['accuracy']
)


#  ---- Этап 3 ----

history1 = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=0
)
# --- Этап 4 ---

plt.plot(history1.history['accuracy'])
plt.show()
# Сравнивая точность для 5 нейронов в скрытом слое и 10ти, 5 скрытых нейронов для одинаковых настроек справились
# Чуть лучше

# -- Этап 5 --
# ฅ(´=ω=)`ฅ
