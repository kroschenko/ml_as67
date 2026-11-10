
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("   Запуск задачи регрессии (Fish Market)")

df_fish = pd.read_csv("Fish.csv")

features = ["Length1", "Length2", "Length3", "Height", "Width"]

X_reg = df_fish[features]
y_reg = df_fish["Weight"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(X_train_r, y_train_r)

y_pred_r = lin_reg.predict(X_test_r)

r2 = r2_score(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))

print("Количество записей:", len(df_fish))
print("Количество признаков:", X_reg.shape[1])
print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(8, 6))

plt.scatter(
    df_fish["Length3"],
    df_fish["Weight"],
    alpha=0.6,
    color="blue",
    label="Данные"
)

m, b = np.polyfit(
    df_fish["Length3"],
    df_fish["Weight"],
    1
)

x_line = np.linspace(
    df_fish["Length3"].min(),
    df_fish["Length3"].max(),
    100
)

plt.plot(
    x_line,
    m * x_line + b,
    color="red",
    label="Линия регрессии"
)

plt.title("Зависимость веса рыбы от Length3")
plt.xlabel("Length3")
plt.ylabel("Weight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("\n   Запуск задачи классификации (Bank Marketing)")

df_bank = pd.read_csv("bank.csv", sep=";")

X_clf = df_bank.drop("y", axis=1)
y_clf = (df_bank["y"] == "yes").astype(int)

print("Количество записей:", len(df_bank))
print("Количество признаков:", X_clf.shape[1])
print("Клиенты без подписки:", (y_clf == 0).sum())
print("Клиенты с подпиской:", (y_clf == 1).sum())

categorical_features = X_clf.select_dtypes(
    exclude=["number"]
).columns.tolist()

X_clf = pd.get_dummies(
    X_clf,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf
)

scaler = StandardScaler()

X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

log_reg = LogisticRegression(
    random_state=42,
    max_iter=2000
)

log_reg.fit(X_train_c, y_train_c)

y_pred_c = log_reg.predict(X_test_c)

accuracy = accuracy_score(y_test_c, y_pred_c)
precision = precision_score(y_test_c, y_pred_c)
recall = recall_score(y_test_c, y_pred_c)

cm = confusion_matrix(y_test_c, y_pred_c)

print("\nРЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("\nМАТРИЦА ОШИБОК")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nАНАЛИЗ МАТРИЦЫ ОШИБОК")
print(f"Истинно отрицательные (TN): {tn}")
print(f"Ложноположительные (FP): {fp}")
print(f"Ложноотрицательные (FN): {fn}")
print(f"Истинно положительные (TP): {tp}")
print(f"Сумма элементов матрицы: {cm.sum()}")

plt.figure(figsize=(6, 5))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap=plt.cm.Blues
)

plt.title("Матрица ошибок — Bank Marketing")
plt.colorbar()

tick_marks = np.arange(2)

plt.xticks(tick_marks, ["No", "Yes"])
plt.yticks(tick_marks, ["No", "Yes"])

thresh = cm.max() / 2.0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

plt.xlabel("Предсказанные значения")
plt.ylabel("Истинные значения")
plt.tight_layout()
plt.show()
