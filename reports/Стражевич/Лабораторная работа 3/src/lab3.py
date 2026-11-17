
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# 1. Загрузка датасета Seeds

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"

columns = [
    "Area", "Perimeter", "Compactness",
    "Length", "Width", "Asymmetry",
    "Groove", "Class"
]

data = pd.read_csv(url, sep=r"\s+", header=None, names=columns)

# Названия сортов пшеницы
varieties = {
    1: "Kama",
    2: "Rosa",
    3: "Canadian"
}

data["Class"] = data["Class"].map(varieties)

print("Первые 5 строк датасета:")
print(data.head())

print("\nКоличество объектов каждого сорта:")
print(data["Class"].value_counts())


# 2. Разделение признаков и целевой переменной

X = data.drop("Class", axis=1)
y = data["Class"]

# Делим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Стандартизация данных

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 4. Исследование модели k-NN при разных k

k_values = [1, 3, 5, 7, 9]
knn_results = {}

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    knn_results[k] = accuracy

    print(f"k = {k}, Accuracy = {accuracy:.4f}")

# Выбираем лучшее значение k
best_k = max(knn_results, key=knn_results.get)

print("\nЛучшее значение k:", best_k)


# 5. Обучение трех классификаторов

# k-NN
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Дерево решений
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Метод опорных векторов
svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)


# 6. Оценка точности моделей

models = {
    "k-NN": knn,
    "Decision Tree": tree,
    "SVM": svm
}

results = {}

for name, model in models.items():
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"{name}: Accuracy = {accuracy:.4f}")


# Определяем лучшую модель
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nЛучшая модель:", best_model_name)


# 7. График зависимости точности k-NN от количества соседей

plt.figure(figsize=(7, 4))

plt.plot(
    k_values,
    list(knn_results.values()),
    marker="o"
)

plt.xlabel("Количество соседей (k)")
plt.ylabel("Accuracy")
plt.title("Зависимость точности k-NN от значения k")
plt.xticks(k_values)
plt.grid()
plt.show()


# 8. Сравнение точности трех моделей

plt.figure(figsize=(7, 4))

plt.bar(results.keys(), results.values())

plt.xlabel("Модели")
plt.ylabel("Accuracy")
plt.title("Сравнение точности классификаторов")
plt.ylim(0, 1.05)
plt.grid(axis="y")
plt.show()


# 9. Визуализация данных с помощью PCA

# Преобразуем данные из 7 измерений в 2
pca = PCA(n_components=2)

pca.fit(X_train)

X_test_pca = pca.transform(X_test)

# Получаем предсказания лучшей модели
best_predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 5))

for variety in varieties.values():
    mask = best_predictions == variety

    plt.scatter(
        X_test_pca[mask, 0],
        X_test_pca[mask, 1],
        label=variety
    )

plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.title("Классификация семян пшеницы с помощью PCA")
plt.legend(title="Сорта пшеницы")
plt.grid()
plt.show()


# 10. Итоговые результаты

print("\nИТОГОВОЕ СРАВНЕНИЕ:")

for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")

print("\nЛучшая модель:", best_model_name)
print("Точность:", round(results[best_model_name], 4))