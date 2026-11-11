
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

torch.manual_seed(42)
np.random.seed(42)

print("   Запуск задачи классификации (Breast Cancer Wisconsin)")

data = pd.read_csv("breast_cancer.csv")

data = data.drop(
    columns=["id", "Unnamed: 32"],
    errors="ignore"
)

X = data.drop("diagnosis", axis=1).values
y = (data["diagnosis"] == "M").astype(int).values

print("Количество записей:", len(data))
print("Количество признаков:", X.shape[1])
print("Доброкачественные опухоли:", np.sum(y == 0))
print("Злокачественные опухоли:", np.sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(
    y_train.reshape(-1, 1),
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test.reshape(-1, 1),
    dtype=torch.float32
)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def train_model(hidden_size, name):
    torch.manual_seed(42)

    model = MLP(X_train.shape[1], hidden_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(1, 201):
        model.train()

        output = model(X_train)
        loss = criterion(output, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"{name} | Эпоха {epoch}, Loss: {loss.item():.4f}")

    model.eval()

    with torch.no_grad():
        output = model(X_test)
        predictions = (output >= 0.5).int().numpy().flatten()

    actual = y_test.numpy().flatten()

    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions)
    recall = recall_score(actual, predictions)
    f1 = f1_score(actual, predictions)

    matrix = confusion_matrix(actual, predictions)

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nМатрица ошибок:")
    print(matrix)

    return {
        "Модель": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


print("\nОБУЧЕНИЕ МОДЕЛИ С 16 НЕЙРОНАМИ")
result_16 = train_model(16, "MLP с 16 нейронами")

print("\nОБУЧЕНИЕ МОДЕЛИ С 32 НЕЙРОНАМИ")
result_32 = train_model(32, "MLP с 32 нейронами")

print("\nСРАВНЕНИЕ РЕЗУЛЬТАТОВ")

results = pd.DataFrame([result_16, result_32])

print(results.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

print("\nИЗМЕНЕНИЕ МЕТРИК")

precision_change = result_32["Precision"] - result_16["Precision"]
recall_change = result_32["Recall"] - result_16["Recall"]

print(f"Изменение Precision: {precision_change:+.4f}")
print(f"Изменение Recall: {recall_change:+.4f}")

if precision_change > 0:
    print("Precision увеличилась.")
elif precision_change < 0:
    print("Precision уменьшилась.")
else:
    print("Precision не изменилась.")

if recall_change > 0:
    print("Recall увеличилась.")
elif recall_change < 0:
    print("Recall уменьшилась.")
else:
    print("Recall не изменилась.")
