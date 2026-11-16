
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("   Запуск задачи классификации (Telco Customer Churn)")

df = pd.read_csv("Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

X = df.drop(["customerID", "Churn"], axis=1)
y = (df["Churn"] == "Yes").astype(int)

print("Количество записей:", len(df))
print("Количество признаков:", X.shape[1])
print("Клиенты без оттока:", (y == 0).sum())
print("Клиенты с оттоком:", (y == 1).sum())

print("Пропуски TotalCharges:", X["TotalCharges"].isna().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

for column in numeric_features:
    median = X_train[column].median()

    X_train.loc[:, column] = X_train[column].fillna(median)
    X_test.loc[:, column] = X_test[column].fillna(median)

for column in categorical_features:
    mode = X_train[column].mode()[0]

    X_train.loc[:, column] = X_train[column].fillna(mode)
    X_test.loc[:, column] = X_test[column].fillna(mode)

X_train = pd.get_dummies(
    X_train,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale")
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    results.append({
        "Модель": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

results = pd.DataFrame(results).sort_values(
    by="F1-score",
    ascending=False
)

print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
print(results.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

print("\nЛУЧШАЯ МОДЕЛЬ")
print(results.iloc[0]["Модель"])
print(f"F1-score: {results.iloc[0]['F1-score']:.4f}")
