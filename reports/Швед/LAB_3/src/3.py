import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Загрузка и знакомство с данными
df = pd.read_csv(os.path.join(BASE_DIR, 'iris.csv'))

print("Первые 5 строк:")
print(df.head())
print(f"\nРазмер датасета: {df.shape}")
print(f"Пропусков: {df.isnull().sum().sum()}")
print("\nРаспределение классов:")
print(df['variety'].value_counts())
print("\nОписательная статистика:")
print(df.describe().round(2))

plt.figure(figsize=(8, 6))
for name, group in df.groupby('variety'):
    plt.scatter(group['petal.length'], group['petal.width'], label=name, alpha=0.7)
plt.title('Ирисы: длина и ширина лепестка')
plt.xlabel('petal.length')
plt.ylabel('petal.width')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'iris_scatter.png'), dpi=150)
plt.show()

X = df.drop('variety', axis=1)
y = df['variety']

# 2. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nОбучающая выборка: {len(X_train)}, тестовая: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Подбор k для k-NN (кросс-валидация на обучающей выборке)
k_values = range(1, 31)
cv_scores = []
test_scores = []
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    cv_scores.append(cross_val_score(knn_k, X_train_scaled, y_train, cv=5).mean())
    knn_k.fit(X_train_scaled, y_train)
    test_scores.append(accuracy_score(y_test, knn_k.predict(X_test_scaled)))

best_k = list(k_values)[cv_scores.index(max(cv_scores))]
print("\nk-NN: точность в зависимости от k")
print(" k  | CV accuracy | Test accuracy")
for k, cv, test in zip(k_values, cv_scores, test_scores):
    print(f"{k:3d} |   {cv:.4f}    |   {test:.4f}")
print(f"Оптимальное k (по кросс-валидации): {best_k}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o', label='Кросс-валидация (обучающая выборка)')
plt.plot(k_values, test_scores, marker='s', label='Тестовая выборка')
plt.axvline(best_k, color='red', linestyle='--', label=f'Оптимальное k = {best_k}')
plt.title('Точность k-NN в зависимости от количества соседей k')
plt.xlabel('k (количество соседей)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'knn_k.png'), dpi=150)
plt.show()

# 4. Обучение трёх моделей
knn = KNeighborsClassifier(n_neighbors=best_k)
dt = DecisionTreeClassifier(random_state=42)
svm = SVC(random_state=42)

knn.fit(X_train_scaled, y_train)
dt.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
y_pred_dt = dt.predict(X_test_scaled)
y_pred_svm = svm.predict(X_test_scaled)

# 5. Оценка точности на тестовой выборке
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_svm = accuracy_score(y_test, y_pred_svm)

print(f"\nAccuracy k-NN (k={best_k}): {acc_knn:.4f}")
print(f"Accuracy Decision Tree: {acc_dt:.4f}")
print(f"Accuracy SVM: {acc_svm:.4f}")

# Дополнительно: кросс-валидация на всей выборке (устойчивее для 150 объектов)
print("\nСредняя точность по 5-кратной кросс-валидации (вся выборка):")
X_all_scaled = StandardScaler().fit_transform(X)
for name, model in [(f'k-NN (k={best_k})', KNeighborsClassifier(n_neighbors=best_k)),
                    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
                    ('SVM', SVC(random_state=42))]:
    scores = cross_val_score(model, X_all_scaled, y, cv=5)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# 6. Сравнение
results = {f'k-NN (k={best_k})': acc_knn, 'Decision Tree': acc_dt, 'SVM': acc_svm}

plt.figure(figsize=(8, 5))
bars = plt.bar(results.keys(), results.values(), color=['steelblue', 'seagreen', 'darkorange'])
for bar, acc in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.005, f'{acc:.3f}', ha='center')
plt.ylim(0.8, 1.02)
plt.title('Сравнение точности моделей на тестовой выборке')
plt.ylabel('Accuracy')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'models_comparison.png'), dpi=150)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, y_pred) in zip(axes, [(f'k-NN (k={best_k})', y_pred_knn),
                                     ('Decision Tree', y_pred_dt), ('SVM', y_pred_svm)]):
    cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
    ConfusionMatrixDisplay(cm, display_labels=knn.classes_).plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrices.png'), dpi=150)
plt.show()

best_model = max(results.items(), key=lambda x: x[1])[0]
print(f"\nСамый эффективный метод на данном датасете - {best_model}")
