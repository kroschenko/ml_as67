import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Честно, я смирился с ошибками, пусть будут
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Такс
# 1, Значит, как в прошлый раз, первый этап, это загрузить данные и обработать их
# 2, Далее создать модель логистической регрессии
# 3, Обучаем (что можно) и смотрим результаты
# 3.5, если результат не устраивает, перехожу к пункту 2 и делаю другую модель, если всё гуд, то иду к 4ому пункту
# 4, Строю матрицу ошибок (сразу после того, как пойму точно, что это)
# 5, Всё готово


# Окей, настраиваемся
SEED = 44
matplotlib.use('TkAgg')
# Добавим крутую сеточку в графики
sns.set_style("whitegrid")


# ----- Пункт 1 -----
# Расчехляем наш датасет
df = pd.read_csv("Titanic-Dataset.csv")
# А вообще, я просто с первой лабы возьму всё, просто уберу лишние поля и разделю на три выборки
# Заполнение пропусков в Age медианным значением и взятие данных для гистограммы
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# One-Hot Encoding для Sex и Embarked
df = pd.get_dummies(df, columns=["Sex", "Embarked"], prefix=["Sex", "Embarked"])

# https://habr.com/ru/companies/ruvds/articles/494720/ Шпоры
# Так, тут важный момент. Удаляю Кабины, так как куча пропусков, да и информация сомнительной полезности
# Номер билета тоже ни о чём не будет говорить для нейросети, лишняя информация. Имя и айди тоже
df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)

# Далее, берём и перемешиваем инфу.
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# Теперь разбиваем данные две выборки, тестовую и тренировочную
n = len(df)
n_train = int(n * 0.90)

x_all = df.drop("Survived", axis=1).values.astype(np.float32)
y_all = df["Survived"].values.astype(np.float32)

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
    keras.layers.Dense(2),
    keras.layers.Dense(1, activation="sigmoid")
])

# Правила обучения
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    # Бинарная кросс энтропия, лучший выбор для категорирования между двумя вариантами, как в данном случае
    loss="binary_crossentropy",
    metrics=['accuracy']
)


#  ---- Этап 3 ----
# Обучаем модель и смотрим, на сколько маленькая ошибка
# Я кста узнал, что можно не собственноручно разбивать на валидационную и обучающую
# Ну а так как доступ к конкретным экземплярам валидационной выборки мне не нужны, то сделаем так

# Так, появляется ошибка обучения, цель и выход должны иметь одинаковый ранг ???? Непанятн
history1 = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=0
)
# --- Этап 3.5 ---

plt.plot(history1.history['accuracy'])
plt.show()

# --- Этап 4 ---
# И так, всё супер, теперь матрица ошибок

y_prob = model.predict(x_test, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("\nМатрица ошибок (строки — истина, столбцы — предсказание):")
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=["Погиб (0)", "Выжил (1)"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Матрица ошибок — Titanic")
plt.tight_layout()
plt.show()
# Ну что же, честно говоря, я действительно не ожидал, что будет достигнута такая точность 0_о.
# Честно говоря, пока я эту матрицу не вывел, я действительно преуменьшал точность нейросети =/

# -- Этап 5 --
# Блин, я так задолбался, только этот смайл опишет моё состояние:
# ฅ(´-ω-)`ฅ
