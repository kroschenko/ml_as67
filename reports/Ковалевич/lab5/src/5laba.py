import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

W, H = 8, 3
b, c, d = 0.5, 0.09, 0.5
A_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]
EPOCHS = 3000
LR = 0.01
TRAIN_SHARE = 0.7

def make_data(a):
    x = np.arange(0, 60, 0.5)
    y = a * np.cos(b * x) + c * np.sin(d * x)
    X = np.array([y[i:i + W] for i in range(len(y) - W)])
    T = y[W:]
    n_train = int(len(X) * TRAIN_SHARE)
    return x, y, X, T, n_train

def train_model(Xtr, Ttr):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(W, H), nn.Sigmoid(), nn.Linear(H, 1))
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Tt = torch.tensor(Ttr, dtype=torch.float32).view(-1, 1)
    losses = []
    for _ in range(EPOCHS):
        loss = loss_fn(model(Xt), Tt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses

def predict(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()


def run(a):
    x, y, X, T, n_train = make_data(a)

    lo, hi = y[:n_train + W].min(), y[:n_train + W].max()
    sc = lambda v: (v - lo) / (hi - lo)
    unsc = lambda v: v * (hi - lo) + lo

    Xs = sc(X)
    model, losses = train_model(Xs[:n_train], sc(T)[:n_train])
    losses = np.array(losses) * (hi - lo) ** 2

    train_pred = unsc(predict(model, Xs[:n_train]))
    train_ref = T[:n_train]

    n_test = len(T) - n_train
    window = list(Xs[n_train])
    test_pred_s = []
    for _ in range(n_test):
        p = predict(model, np.array([window]))[0]
        test_pred_s.append(p)
        window = window[1:] + [p]
    test_pred = unsc(np.array(test_pred_s))
    test_ref = T[n_train:]

    return dict(a=a, x=x, y=y, n_train=n_train, losses=losses,
                train_ref=train_ref, train_pred=train_pred,
                test_ref=test_ref, test_pred=test_pred,
                mse_train=np.mean((train_ref - train_pred) ** 2),
                mse_test=np.mean((test_ref - test_pred) ** 2))

results = [run(a) for a in A_LIST]

summary = pd.DataFrame({
    "a": [r["a"] for r in results],
    "MSE обучения": [r["mse_train"] for r in results],
    "MSE прогноза": [r["mse_test"] for r in results],
})
print(summary.to_string(index=False))

best = min(results, key=lambda r: r["mse_test"])
print(f"\nЛучшее a = {best['a']}  (MSE прогноза = {best['mse_test']:.3e})")

a_best, n_train = best["a"], best["n_train"]
x, y = best["x"], best["y"]

tab_train = pd.DataFrame({
    "Эталонное значение": best["train_ref"],
    "Полученное значение": best["train_pred"],
})
tab_train["Отклонение"] = tab_train["Эталонное значение"] - tab_train["Полученное значение"]

tab_test = pd.DataFrame({
    "Эталонное значение": best["test_ref"],
    "Полученное значение": best["test_pred"],
})
tab_test["Отклонение"] = tab_test["Эталонное значение"] - tab_test["Полученное значение"]

print("\nТаблица обучения:\n", tab_train.to_string())
print("\nТаблица прогноза:\n", tab_test.to_string())

plt.figure(figsize=(9, 4))
plt.plot(x[:n_train + W], y[:n_train + W], label="y = a·cos(bx) + c·sin(dx)")
plt.title(f"Прогнозируемая функция на участке обучения (a = {a_best})")
plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend()
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(best["losses"])
plt.yscale("log")
plt.title("Изменение ошибки (MSE) в зависимости от итерации")
plt.xlabel("Итерация (эпоха)"); plt.ylabel("MSE"); plt.grid(True)
plt.show()

plt.figure(figsize=(9, 4))
xt = x[W:W + n_train]
xp = x[W + n_train:]
plt.plot(xt, best["train_ref"], label="эталон (обучение)")
plt.plot(xt, best["train_pred"], "--", label="ИНС (обучение)")
plt.plot(xp, best["test_ref"], label="эталон (прогноз)")
plt.plot(xp, best["test_pred"], "--", label="ИНС (прогноз)")
plt.title(f"Результаты обучения и прогнозирования (a = {a_best})")
plt.xlabel("x"); plt.ylabel("y"); plt.grid(True); plt.legend()
plt.show()