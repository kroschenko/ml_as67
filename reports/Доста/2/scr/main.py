import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)


#1 РЕГРЕССИЯ
print("РЕГРЕССИЯ: Medical Cost")

med = pd.read_csv("medical_cost_personal_dataset.csv")

med = pd.get_dummies(med, columns=["sex", "smoker", "region"], drop_first=True)

X_reg = med.drop(columns=["charges"])
y_reg = med["charges"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2
)

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")
print(f"R²:  {r2_score(y_test_reg, y_pred_reg):.4f}")

simple_bmi_model = LinearRegression().fit(med[["bmi"]], med["charges"])
x_line = pd.DataFrame({"bmi": np.linspace(med["bmi"].min(), med["bmi"].max(), 200)})
y_line = simple_bmi_model.predict(x_line)

plt.figure(figsize=(8, 5))
plt.scatter(med["bmi"], med["charges"], alpha=0.6)
plt.plot(x_line["bmi"], y_line, color="red", linewidth=2, label="Линия регрессии")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.title("Зависимость charges от bmi")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


#2 КЛАССИФИКАЦИЯ
print("КЛАССИФИКАЦИЯ: Heart Disease UCI")

heart = pd.read_csv("heart_disease_uci.csv")

heart = heart[heart["num"].notna()].copy()
heart["target"] = (heart["num"] > 0).astype(int)
heart = heart.drop(columns=["id", "num", "dataset"])

for col in heart.select_dtypes(include=[np.number]).columns:
    heart[col] = heart[col].fillna(heart[col].median())

for col in heart.select_dtypes(exclude=[np.number]).columns:
    heart[col] = heart[col].fillna(heart[col].mode()[0])

heart = pd.get_dummies(heart, drop_first=True)

X_clf = heart.drop(columns=["target"])
y_clf = heart["target"]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, stratify=y_clf
)

scaler = StandardScaler()
X_train_clf = scaler.fit_transform(X_train_clf)
X_test_clf = scaler.transform(X_test_clf)

clf_model = LogisticRegression(max_iter=2000)
clf_model.fit(X_train_clf, y_train_clf)
y_pred_clf = clf_model.predict(X_test_clf)

print(f"Accuracy:  {accuracy_score(y_test_clf, y_pred_clf):.4f}")
print(f"Precision: {precision_score(y_test_clf, y_pred_clf):.4f}")
print(f"Recall:    {recall_score(y_test_clf, y_pred_clf):.4f}")
print(f"F1-score:  {f1_score(y_test_clf, y_pred_clf):.4f}")

print("\nClassification report:")
print(classification_report(y_test_clf, y_pred_clf, target_names=["Нет болезни", "Болезнь"]))

cm = confusion_matrix(y_test_clf, y_pred_clf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Нет болезни", "Болезнь"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Матрица ошибок — Heart Disease UCI")
plt.show()