import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

A, B, C, D = 0.1, 0.1, 0.05, 0.1
N_IN, N_HID = 6, 2
HORIZON = 20

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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_IN, N_HID)
        self.fc2 = nn.Linear(N_HID, 1)
    def forward(self, x):
        return self.fc2(torch.sigmoid(self.fc1(x)))

def train(alpha, X, Y, epochs=2000):
    model = Net()
    opt = torch.optim.SGD(model.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), Yt)
        loss.backward()
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

with torch.no_grad():
    ptr = inv(model(torch.tensor(Xtr)).numpy().ravel())
ttr = inv(Ytr.ravel())

train_df = pd.DataFrame({"Эталон": ttr, "Получено": ptr, "Отклонение": ptr - ttr})
print("\nОбучение, первые 10 строк:")
print(train_df.head(10).to_string(index=False))
train_df.to_csv("train.csv", index=False)

with torch.no_grad():
    pte = inv(model(torch.tensor(Xte)).numpy().ravel())
tte = inv(Yte.ravel())

test_df = pd.DataFrame({"Эталон": tte, "Получено": pte, "Отклонение": pte - tte})
print("\nТест, первые 10 строк:")
print(test_df.head(10).to_string(index=False))
test_df.to_csv("test.csv", index=False)

print(f"\nRMSE={np.sqrt(np.mean((pte-tte)**2)):.6f}, MAE={np.mean(np.abs(pte-tte)):.6f}")

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Итерация"); plt.ylabel("MSE")
plt.title(f"Ошибка обучения, alpha={best_alpha}")
plt.grid(); plt.tight_layout()
plt.savefig("loss.png", dpi=150); plt.show()

plt.figure(figsize=(10, 4))
plt.plot(ttr, label="Эталон")
plt.plot(ptr, label="ИНС")
plt.xlabel("Номер окна"); plt.ylabel("y")
plt.title("Прогноз на участке обучения")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("train.png", dpi=150); plt.show()

window = ys[n_train:n_train+N_IN].copy()
fc = []
model.eval()
for _ in range(HORIZON):
    with torch.no_grad():
        v = model(torch.tensor(window.reshape(1, -1), dtype=torch.float32)).item()
    fc.append(inv(v))
    window = np.append(window[1:], v)

true_fc = y[n_train+N_IN : n_train+N_IN+HORIZON]
fc_df = pd.DataFrame({"Эталон": true_fc, "Получено": fc,
                      "Отклонение": np.array(fc) - true_fc})
print("\nМногошаговый прогноз:")
print(fc_df.to_string(index=False))
fc_df.to_csv("forecast.csv", index=False)

plt.figure(figsize=(10, 4))
plt.plot(true_fc, marker="o", label="Эталон")
plt.plot(fc, marker="s", label="ИНС")
plt.xlabel("Шаг"); plt.ylabel("y")
plt.title("Многошаговый прогноз")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("forecast.png", dpi=150); plt.show()