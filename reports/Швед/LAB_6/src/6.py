import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Вариант 1: y = a*cos(b*x) + c*sin(d*x), 6 входов, 2 НЭ в скрытом слое, РНС Элмана
A, B, C, D = 0.1, 0.1, 0.05, 0.1
N_IN, N_HID = 6, 2
HORIZON = 20
torch.manual_seed(42)

x = np.arange(0, 20.1, 0.1)
y = A*np.cos(B*x) + C*np.sin(D*x)

y_min, y_max = y.min(), y.max()
ys = (y - y_min) / (y_max - y_min)
inv = lambda v: v*(y_max - y_min) + y_min

X, Y = [], []
for i in range(len(ys) - N_IN):
    X.append(ys[i:i+N_IN])
    Y.append(ys[i+N_IN])
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32).reshape(-1, 1)

n_train = int(0.8 * len(X))
Xtr, Ytr = X[:n_train], Y[:n_train]
Xte, Yte = X[n_train:], Y[n_train:]


class Elman(nn.Module):
    """РНС Элмана: h(t) = sigmoid(W_in*x(t) + W_ctx*h(t-1) + b), y(t) = W_out*h(t) + b_out.
    Контекстный слой хранит выход скрытого слоя на предыдущем шаге."""
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(N_IN, N_HID)                 # вход -> скрытый слой
        self.ctx = nn.Linear(N_HID, N_HID, bias=False)    # контекст -> скрытый слой
        self.out = nn.Linear(N_HID, 1)                    # скрытый слой -> выход (линейный)

    def forward(self, X, h=None):
        # X - последовательность окон в порядке времени, форма (T, N_IN)
        if h is None:
            h = torch.zeros(N_HID)
        outputs = []
        for x_t in X:
            h = torch.sigmoid(self.inp(x_t) + self.ctx(h))
            outputs.append(self.out(h))
        return torch.stack(outputs), h


def train(alpha, X, Y, epochs=2000):
    model = Elman()
    opt = torch.optim.SGD(model.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        pred, _ = model(Xt)                  # проход по всей обучающей последовательности
        loss = loss_fn(pred, Yt)
        loss.backward()                      # обратное распространение во времени (BPTT)
        opt.step()
        losses.append(loss.item())
    return model, losses


results = []
for a in [0.01, 0.05, 0.1, 0.3, 0.5]:
    _, L = train(a, Xtr, Ytr, epochs=1000)
    results.append((a, L[-1]))
    print(f"alpha={a}, MSE={L[-1]:.6f}")

best_alpha = min(results, key=lambda t: t[1])[0]
print("Лучшее alpha:", best_alpha)

model, losses = train(best_alpha, Xtr, Ytr, epochs=5000)
print(f"MSE после 5000 итераций: {losses[-1]:.6f}")

model.eval()
with torch.no_grad():
    out_tr, h_train = model(torch.tensor(Xtr))
    ptr = inv(out_tr.numpy().ravel())
    # тест: продолжаем последовательность с состоянием контекста после обучающего участка
    out_te, _ = model(torch.tensor(Xte), h_train)
    pte = inv(out_te.numpy().ravel())
ttr = inv(Ytr.ravel())
tte = inv(Yte.ravel())

train_df = pd.DataFrame({"Эталон": ttr, "Получено": ptr, "Отклонение": ptr - ttr})
print("\nОбучение, первые 10 строк:")
print(train_df.head(10).to_string(index=False))
train_df.to_csv(os.path.join(BASE_DIR, "train.csv"), index=False)
print(f"Обучение: RMSE={np.sqrt(np.mean((ptr-ttr)**2)):.6f}, MAE={np.mean(np.abs(ptr-ttr)):.6f}")

test_df = pd.DataFrame({"Эталон": tte, "Получено": pte, "Отклонение": pte - tte})
print("\nТест, первые 10 строк:")
print(test_df.head(10).to_string(index=False))
test_df.to_csv(os.path.join(BASE_DIR, "test.csv"), index=False)
print(f"Тест: RMSE={np.sqrt(np.mean((pte-tte)**2)):.6f}, MAE={np.mean(np.abs(pte-tte)):.6f}")

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Итерация"); plt.ylabel("MSE")
plt.title(f"Ошибка обучения, alpha={best_alpha}")
plt.grid(); plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "loss.png"), dpi=150); plt.show()

plt.figure(figsize=(10, 4))
plt.plot(ttr, label="Эталон")
plt.plot(ptr, label="РНС Элмана")
plt.xlabel("Номер окна"); plt.ylabel("y")
plt.title("Прогноз на участке обучения")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "train.png"), dpi=150); plt.show()

# Многошаговый прогноз: каждое предсказание подаётся обратно на вход,
# состояние контекстного слоя переносится от шага к шагу
window = ys[n_train:n_train+N_IN].copy()
h = h_train
fc = []
with torch.no_grad():
    for _ in range(HORIZON):
        out, h = model(torch.tensor(window.reshape(1, -1), dtype=torch.float32), h)
        v = out.item()
        fc.append(inv(v))
        window = np.append(window[1:], v)

true_fc = y[n_train+N_IN: n_train+N_IN+HORIZON]
fc_df = pd.DataFrame({"Эталон": true_fc, "Получено": fc,
                      "Отклонение": np.array(fc) - true_fc})
print("\nМногошаговый прогноз:")
print(fc_df.to_string(index=False))
fc_df.to_csv(os.path.join(BASE_DIR, "forecast.csv"), index=False)
print(f"Прогноз: RMSE={np.sqrt(np.mean((np.array(fc)-true_fc)**2)):.6f}, "
      f"MAE={np.mean(np.abs(np.array(fc)-true_fc)):.6f}")

plt.figure(figsize=(10, 4))
plt.plot(true_fc, marker="o", label="Эталон")
plt.plot(fc, marker="s", label="РНС Элмана")
plt.xlabel("Шаг"); plt.ylabel("y")
plt.title("Многошаговый прогноз")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "forecast.png"), dpi=150); plt.show()
