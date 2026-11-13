
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

torch.manual_seed(42)
np.random.seed(42)

print("   Запуск задачи классификации (Digits)")

digits = load_digits()

X = digits.data
y = digits.target

print("Количество записей:", len(X))
print("Количество признаков:", X.shape[1])
print("Количество классов:", len(np.unique(y)))

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

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, input_size, second_layer=False):
        super().__init__()

        if second_layer:
            self.layers = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )

    def forward(self, x):
        return self.layers(x)


def train_model(second_layer, name):
    torch.manual_seed(42)

    model = MLP(X_train.shape[1], second_layer)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        probabilities = torch.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1).numpy()

    actual = y_test.numpy()

    accuracy = accuracy_score(actual, predictions)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


print("\nОБУЧЕНИЕ МОДЕЛИ С ОДНИМ СКРЫТЫМ СЛОЕМ")
accuracy_1 = train_model(
    False,
    "MLP с одним скрытым слоем"
)

print("\nОБУЧЕНИЕ МОДЕЛИ С ДВУМЯ СКРЫТЫМИ СЛОЯМИ")
accuracy_2 = train_model(
    True,
    "MLP с двумя скрытыми слоями"
)

print("\nСРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print(f"Один скрытый слой:  {accuracy_1:.4f}")
print(f"Два скрытых слоя:   {accuracy_2:.4f}")

change = accuracy_2 - accuracy_1

print(f"Изменение Accuracy: {change:+.4f}")

if accuracy_2 > accuracy_1:
    print("Добавление второго скрытого слоя улучшило точность.")
elif accuracy_2 < accuracy_1:
    print("Добавление второго скрытого слоя снизило точность.")
else:
    print("Точность моделей не изменилась.")
