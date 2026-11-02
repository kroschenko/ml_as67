import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# ЧАСТЬ 1. РЕГРЕССИЯ — BOSTON HOUSING
# ============================================================
print("=" * 70)
print("ЧАСТЬ 1. ЛИНЕЙНАЯ РЕГРЕССИЯ — BOSTON HOUSING")
print("=" * 70)

boston = pd.read_csv("BostonHousing.csv")
# В файле есть категориальная копия цели — для регрессии её убираем
if "CAT. MEDV" in boston.columns:
    boston = boston.drop(columns=["CAT. MEDV"])

y_reg = boston["MEDV"]
X_reg = boston.drop(columns=["MEDV"])

print("Размер данных Boston:", boston.shape)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lin_reg = LinearRegression().fit(X_train_reg, y_train_reg)
y_pred_reg = lin_reg.predict(X_test_reg)

print(f"MSE = {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
print(f"R²  = {r2_score(y_test_reg, y_pred_reg):.4f}")

# ---- График MEDV от RM с линией регрессии ----
rm = boston[["RM"]]
medv = boston["MEDV"]

simple_reg = LinearRegression().fit(rm, medv)

# Чтобы не было предупреждения о feature names, предсказываем по DataFrame
rm_line = pd.DataFrame(
    {"RM": np.linspace(rm["RM"].min(), rm["RM"].max(), 100)}
)
medv_line = simple_reg.predict(rm_line)

plt.figure(figsize=(8, 6))
plt.scatter(rm["RM"], medv, alpha=0.5, color="steelblue", label="Наблюдения")
plt.plot(rm_line["RM"], medv_line, color="red", linewidth=2,
         label=f"MEDV = {simple_reg.intercept_:.2f} + {simple_reg.coef_[0]:.2f}·RM")
plt.xlabel("RM — среднее число комнат")
plt.ylabel("MEDV — медианная стоимость (тыс. $)")
plt.title("Boston Housing: MEDV от RM")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("boston_medv_vs_rm.png", dpi=150)
plt.show()
print("График сохранён: boston_medv_vs_rm.png")


# ============================================================
# ЧАСТЬ 2. КЛАССИФИКАЦИЯ — BREAST CANCER
# ============================================================
print("\n" + "=" * 70)
print("ЧАСТЬ 2. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ — BREAST CANCER")
print("=" * 70)

bc = pd.read_csv("breast_cancer.csv")

# >>> ГЛАВНОЕ ИСПРАВЛЕНИЕ <<<
# В файле лишняя запятая в заголовке → pandas создаёт столбец
# "Unnamed: 32", целиком состоящий из NaN. Удаляем такие столбцы.
bc = bc.dropna(axis=1, how="all")

# id не нужен для модели
if "id" in bc.columns:
    bc = bc.drop(columns=["id"])

print("Размер данных Breast Cancer:", bc.shape)

# M (malignant) → 1, B (benign) → 0
bc["diagnosis"] = bc["diagnosis"].map({"M": 1, "B": 0})

print("\nРаспределение классов:")
print(bc["diagnosis"].value_counts()
        .rename({1: "M (malignant, 1)", 0: "B (benign, 0)"}))

X_cls = bc.drop(columns=["diagnosis"])
y_cls = bc["diagnosis"]

# Стандартизация признаков
scaler = StandardScaler()
X_cls_scaled = scaler.fit_transform(X_cls)

# Train/test со стратификацией по классу
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls_scaled, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_cls, y_train_cls)
y_pred_cls = log_reg.predict(X_test_cls)

acc  = accuracy_score(y_test_cls, y_pred_cls)
prec = precision_score(y_test_cls, y_pred_cls, pos_label=1)
rec  = recall_score(y_test_cls,  y_pred_cls, pos_label=1)

print(f"\nAccuracy  = {acc:.4f}")
print(f"Precision = {prec:.4f}")
print(f"Recall    = {rec:.4f}")

cm = confusion_matrix(y_test_cls, y_pred_cls, labels=[0, 1])
print("\nМатрица ошибок (строки — истина, столбцы — прогноз):")
print(pd.DataFrame(cm,
                   index=["Истина B(0)", "Истина M(1)"],
                   columns=["Прогноз B(0)", "Прогноз M(1)"]))

disp = ConfusionMatrixDisplay(cm, display_labels=["B (0)", "M (1)"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Матрица ошибок — Breast Cancer")
plt.tight_layout()
plt.savefig("breast_cancer_confusion_matrix.png", dpi=150)
plt.show()
print("Матрица ошибок сохранена: breast_cancer_confusion_matrix.png")

tn, fp, fn, tp = cm.ravel()
print("\n" + "-" * 70)
print("ОБЪЯСНЕНИЕ КВАДРАНТОВ МАТРИЦЫ ОШИБОК")
print("-" * 70)
print(f"TN = {tn}: доброкачественные, верно классифицированы как B")
print(f"FP = {fp}: на самом деле M, предсказаны как B "
      f"(пропуск рака — самая опасная ошибка)")
print(f"FN = {fn}: на самом деле B, предсказаны как M (ложная тревога)")
print(f"TP = {tp}: злокачественные, верно классифицированы как M")
print("\nДля диагностики рака критично минимизировать FP → важен Recall по классу M.")