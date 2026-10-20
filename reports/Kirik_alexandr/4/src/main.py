import torch
from torch import nn
from torch import optim

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
print("Платформа обучения: ", device)

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            # Второй слой нейронной сети
            # nn.ReLU(),
            # nn.Linear(in_features=32, out_features=32),
            nn.Linear(in_features=32, out_features=10),
            # nn.Softmax(dim=10) # не применяется для многоклассовой классификации
        )

    def forward(self, x): 
        return self.linear_relu_stack(x)


# Обучение нейронной сети для многоклассовой классификации
def solution() -> None:
    TARGET : str = 'target'
    digits_dataset : pd.DataFrame = pd.read_csv('digits.csv', sep=',')
    y = digits_dataset[TARGET].values.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
                digits_dataset.drop(columns=TARGET),
                torch.as_tensor(y, dtype=torch.long),
                test_size=0.25,
                random_state=42
        )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    mlp = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    
    epochs : int = 50#125
    print("="*45)
    print("\nОбучаемая модель: \n", mlp)
    print("="*45)
    print("\nЦикл обучения модели: ")
    print("\tКоличество эпох: ", epochs)
    print("\tОптимизатор(Adam): ")
    for name, val in optimizer.defaults.items():
        print("\t\t", name, " : ", val)
    print("\tФункция потерь(многоклассовая): ", criterion)
    print("="*45)
    # Цикл обучения нейронной сети
    for _ in range(epochs):
        mlp.train()
        y_pred = mlp(X_train_tensor)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Оценка модели
    print("\nОценка модели нейронной сети: ")
    mlp.eval()
    with torch.no_grad():
        y_pred = mlp(X_test_tensor)
        predicted_classes = torch.argmax(y_pred, dim=1)

        print(classification_report(y_test, predicted_classes))

if __name__ == "__main__":
    solution()