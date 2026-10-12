import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print('   Запуск задачи регрессии (Auto MPG) ')

df_auto = pd.read_csv('auto-mpg.csv')

df_auto['horsepower'] = pd.to_numeric(df_auto['horsepower'], errors='coerce')
df_auto = df_auto.dropna(subset=['mpg', 'cylinders', 'horsepower', 'weight'])

X_reg = df_auto[['cylinders', 'horsepower', 'weight']]
y_reg = df_auto['mpg']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(X_train_r, y_train_r)

y_pred_r = lin_reg.predict(X_test_r)
mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print(f'MSE: {mse:.4f}')
print(f'R^2: {r2:.4f}')

plt.figure(figsize=(8, 6))
plt.scatter(
    df_auto['horsepower'],
    df_auto['mpg'],
    alpha=0.6,
    color='blue',
    label='Данные',
)
m, b = np.polyfit(df_auto['horsepower'], df_auto['mpg'], 1)
plt.plot(
    df_auto['horsepower'],
    m * df_auto['horsepower'] + b,
    color='red',
    label='Линия регрессии',
)
plt.title(
    'Зависимость расхода топлива (mpg) от лошадиных сил (horsepower)',
    fontsize=12,
)
plt.xlabel('Horsepower', fontsize=10)
plt.ylabel('MPG', fontsize=10)
plt.legend()
plt.grid(True)
plt.show()

print('\n   Запуск задачи классификации ')

col_names = [
    'pregnancies',
    'glucose',
    'blood_pressure',
    'skin_thickness',
    'insulin',
    'bmi',
    'pedigree',
    'age',
    'Outcome',
]
df_diabetes = pd.read_csv(
    'pima-indians-diabetes.csv', comment='#', header=None, names=col_names
)

X_clf = df_diabetes.drop('Outcome', axis=1)
y_clf = df_diabetes['Outcome']

model_pipeline = make_pipeline(
    StandardScaler(), LogisticRegression(random_state=42, max_iter=1000)
)

y_pred_c = cross_val_predict(model_pipeline, X_clf, y_clf, cv=10)

accuracy = accuracy_score(y_clf, y_pred_c)
precision = precision_score(y_clf, y_pred_c)
recall = recall_score(y_clf, y_pred_c)
cm = confusion_matrix(y_clf, y_pred_c)

print(f'Accuracy (на всех данных): {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

tn, fp, fn, tp = cm.ravel()
print('\n    Анализ матрицы ошибок (по всем 768 строкам)')
print(f'Истинно отрицательные (TN): {tn}')
print(f'Ложноположительные (FP): {fp}')
print(f'Ложноотрицательные (FN): {fn}')
print(f'Истинно положительные (TP): {tp}')
print(f'Сумма элементов в матрице: {cm.sum()}')

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(
    'Матрица ошибок  — Диагностика диабета', fontsize=12
)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, [0, 1])
plt.yticks(tick_marks, [0, 1])

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], 'd'),
            horizontalalignment='center',
            color='white' if cm[i, j] > thresh else 'black',
        )

plt.xlabel('Предсказанные значения', fontsize=10)
plt.ylabel('Истинные значения', fontsize=10)
plt.tight_layout()
plt.show()