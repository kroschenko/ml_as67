import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

import pandas as pd

np.random.seed(42)
torch.manual_seed(42)

class ForecastingMLP(nn.Module):
    def __init__(self, input_size=6, hidden_size=2, output_size=1):
        super().__init__()
        self.forecasting_mlp = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.forecasting_mlp(x)



WINDOW = 6
def make_windows(series, window):
    X, Y = [], []
    for i in range(len(series) - window):
        X.append(series[i:(i + window)])
        Y.append(series[i + window])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)

def solution() -> None:
    x = np.arange(0, 50, 0.1, dtype=np.float32)
    y = (0.4 * np.cos(0.4 * x) + 0.08 * np.sin(0.4 * x)).astype(np.float32)

    scaler_y = MinMaxScaler(feature_range=(-1, 1))


    train_size : int = int(len(y)*0.8)
    scaler_y.fit(y[:train_size].reshape(-1, 1))
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    X_all, Y_all = make_windows(y_scaled, WINDOW)

    train_end = train_size - WINDOW
    X_train, Y_train = X_all[:train_end], Y_all[:train_end]
    X_test, Y_test = X_all[train_end:], Y_all[train_end:]

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(Y_train).view(-1, 1)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)



    model = ForecastingMLP()
    print("="*30)
    print("Структура модели:")
    print("="*30)
    print(model)
    print("="*30)
    print(f"Процесс обучения модели:")
    print("="*30)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    epochs = 200
    losses : list = []
    for epoch in range(epochs):
        model.train()
        total_loss : float = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        losses.append(total_loss / len(train_ds))
        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch + 1:4d}/{epochs}, Потери = {losses[-1]:.6f}")


    model.eval()
    with torch.no_grad():
        train_pred_scaled = model(torch.from_numpy(X_train)).numpy().ravel()
        test_pred_scaled = model(torch.from_numpy(X_test)).numpy().ravel()


        test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()
        test_true = scaler_y.inverse_transform(Y_test.reshape(-1, 1)).ravel()

        train_pred = scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
        train_true = scaler_y.inverse_transform(Y_train.reshape(-1, 1)).ravel()

        test_rmse = root_mean_squared_error(test_true, test_pred)
        test_r2 = r2_score(test_true, test_pred)

        train_rmse = root_mean_squared_error(train_true, train_pred)
        train_r2 = r2_score(train_true, train_pred)

    print()
    print("Оценка модели по тренировочным данным:")
    print("="*30)
    print(f"RMSE: {train_rmse:.6f}")
    print(f"R2 : {train_r2:.6f}")
    print()
    print("Оценка модели по тестовым данным:")
    print("="*30)
    print(f"RMSE: {test_rmse:.6f}")
    print(f"R2 : {test_r2:.6f}")

    train_df : pd.DataFrame = pd.DataFrame({
        "Эталонное значение" : train_true.ravel(),
        "Полученное значение" : train_pred.ravel(),
        "Отклонение" : train_pred.ravel() - train_true.ravel()
    })
    train_df.to_csv("train.csv", sep=',')

    test_df : pd.DataFrame = pd.DataFrame({
        "Эталонное значение" : test_true.ravel(),
        "Полученное значение" : test_pred.ravel(),
        "Отклонение" : test_pred.ravel() - test_true.ravel()
    })
    test_df.to_csv("test.csv", sep=',')

    print("\nРезультаты обучения:")
    print(train_df.head(15).to_string(index=True))
    print("\nРезультаты предсказания:")
    print(test_df.head(15).to_string(index=True))
    
    plt.figure(figsize=(12, 5))
    plt.plot(train_true, label="Истинные значения", linewidth=2)
    plt.plot(train_pred, label="Прогноз ИНС", linestyle="--")
    plt.title("График y = 0.4cos(0.4x) + 0.08sin(0.4x) на участке обучения")
    plt.xlabel("Номер окна")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(losses, label="Ошибки", linewidth=2)
    plt.title("Изменение ошибки по итерациям")
    plt.xlabel("Итерация")
    plt.ylabel("Ошибка")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    solution()