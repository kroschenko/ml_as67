# И так, первое, что я решаю, это открыть и почитать что это такое.
# И что же я обнаружил?
# На первой страничке, т.е. в хабре, первыми алгоритмами МО идут именно
# метод k-ближайших соседей (k-NN), деревья решений и метод опорных векторов
# Прям в таком же порядке! Мб это связано с их популярностью, востребованностью и тп, нооо.
# Совпадение? Не думаю!

import pandas as pd
import matplotlib
import seaborn as sns
# Для первой модели:
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
# Изначально импортировал для первой и третьей, но, почему-то, третья лучше без стандартизации??
from sklearn.preprocessing import StandardScaler
# Для второй:
from sklearn.tree import DecisionTreeClassifier
# Для третьей:
from sklearn.svm import SVC
# Для сравнения точности:
from sklearn.metrics import accuracy_score

# А вообще, на третью лабу можно было и картинок накинуть, учитывая тему -_-
# И так, план, как всегда, прост:
# Этап 1; Загрузить данные (просто скопировать код из прошлой лабы и немного переделать)
# Этап 2; разобраться в моделях, и создать их
# Этап 3; Как всегда далее обучаем
# Этап 4; Сравниваем
# Этап 5; Сделать вывод
# Этап 6; Profit

# Как всегда, настраиваемся:
SEED = 45
matplotlib.use('TkAgg')
sns.set_style("whitegrid")

# - Этап 1 -
df = pd.read_csv("iris.csv")

features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
x = df[features]
y = df['variety']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=SEED, stratify=y
)


# -- Этап 2 --
# Начну с KNN, вообще, это как бы не обучается. Но "обучение" в виде задания данных сделаю на 3ем этапе
knn = make_pipeline(
    StandardScaler(),                 # k-NN чувствителен к масштабу признаков
    KNeighborsClassifier(
        # В условии просилось рассмотреть как влияет количество соседей на качество. От 3ёх и выше точность max
        n_neighbors=3,
        weights='distance',
        metric='minkowski',
        p=2,
    )
)

# Теперь дерево, вроде суть ясна, но вот реализация "под капотом" не вполне понятна:
tree = DecisionTreeClassifier(
    criterion="gini",           # Можно использовать 'gini' или 'entropy', по формулам я не понял в чём прикол
    max_depth=5,                # Максимальная глубина
    min_samples_split=10,       # минимум в узле для разбиения
    min_samples_leaf=5,         # минимум в листе
    random_state=SEED,
)

# Чтооооо, я думал этот метод выдаст максимум точности. Но при правильной настройке он только смог приблизиться
svm = SVC(
    kernel="rbf",
    C=3.0,
    gamma="scale",
    probability=False,
)

# --- Этап 3 ---
knn.fit(x_train, y_train)
tree.fit(x_train, y_train)
svm.fit(x_train, y_train)


# ---- Этап 4 ----
knn_y_pred = knn.predict(x_test)
tree_y_pred = tree.predict(x_test)
svm_y_pred = svm.predict(x_test)
print(accuracy_score(y_test, knn_y_pred))
print(accuracy_score(y_test, tree_y_pred))
print(accuracy_score(y_test, svm_y_pred))


# ----- Этап 5 -----
# В общем вывод такой. Как ни настраивай эти модели, но более 93.(3)% они выдать не смогут
# Сравнить что подходит лучше в данном случае не представляется возможным
# ...
# Если брать погрешности в orange, то можно сказать, что лучшим методом оказался knn, обладая наилучшей точностью
# Можно сказать, что он подходит для задач категорирования, в случае малых входных данных, как в данном случае
# После него идёт:
# SVM, немного опережая tree


# ------ Этап 6 ------
# (^._.^)ฅ
