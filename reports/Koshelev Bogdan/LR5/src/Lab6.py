
import os

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

a, b, c, d = 0.2, 0.6, 0.05, 0.6
N_IN, N_HID, N_OUT = 10, 4, 1

h = 0.2                                   
x = np.arange(0, 80, h)                   
y = a * np.cos(b * x) + c * np.sin(d * x)


def make_windows(series, n_in):
    """Скользящее окно: n_in предыдущих значений -> следующее значение."""
    X = np.array([series[i:i + n_in] for i in range(len(series) - n_in)])
    T = series[n_in:].reshape(-1, 1)
    return X, T


X_all, T_all = make_windows(y, N_IN)
x_target = x[N_IN:]                       

n_train = int(0.7 * len(X_all))         
X_tr, T_tr = X_all[:n_train], T_all[:n_train]
X_te, T_te = X_all[n_train:], T_all[n_train:]


sigmoid = lambda s: 1.0 / (1.0 + np.exp(-s))

W1 = np.random.uniform(-0.5, 0.5, (N_IN, N_HID))
b1 = np.zeros((1, N_HID))
W2 = np.random.uniform(-0.5, 0.5, (N_HID, N_OUT))
b2 = np.zeros((1, N_OUT))


def forward(X):
    S1 = X @ W1 + b1
    H = sigmoid(S1)                       
    Y = H @ W2 + b2                      
    return H, Y


lr, momentum = 0.05, 0.9
epochs, target_mse = 20000, 1e-8
vW1, vb1, vW2, vb2 = (np.zeros_like(p) for p in (W1, b1, W2, b2))
history = []

for ep in range(1, epochs + 1):
    H, Y = forward(X_tr)
    E = Y - T_tr
    mse = float(np.mean(E ** 2))
    history.append(mse)
    if mse < target_mse:
        break

    m = len(X_tr)
    dY = 2 * E / m                       
    gW2 = H.T @ dY
    gb2 = dY.sum(axis=0, keepdims=True)
    dH = dY @ W2.T * H * (1 - H)          
    gW1 = X_tr.T @ dH
    gb1 = dH.sum(axis=0, keepdims=True)

    vW1 = momentum * vW1 - lr * gW1
    vb1 = momentum * vb1 - lr * gb1
    vW2 = momentum * vW2 - lr * gW2
    vb2 = momentum * vb2 - lr * gb2
    W1 += vW1
    b1 += vb1
    W2 += vW2
    b2 += vb2


_, Y_tr = forward(X_tr)
_, Y_te = forward(X_te)
mse_tr = np.mean((Y_tr - T_tr) ** 2)
mse_te = np.mean((Y_te - T_te) ** 2)
print(f"Эпох обучения: {ep}")
print(f"СКО (MSE) на обучающей выборке: {mse_tr:.3e}")
print(f"СКО (MSE) на тестовой выборке:  {mse_te:.3e}")
print(f"Макс. абс. ошибка на тесте:      {np.max(np.abs(Y_te - T_te)):.3e}")

print("\nВеса скрытого слоя W1 (10x4):\n", np.round(W1, 4))
print("Смещения скрытого слоя b1:\n", np.round(b1, 4))
print("Веса выходного слоя W2 (4x1):\n", np.round(W2, 4))
print("Смещение выходного слоя b2:\n", np.round(b2, 4))


fig, ax = plt.subplots(3, 1, figsize=(10, 11))

ax[0].semilogy(history)
ax[0].set_title("Процесс обучения ИНС 10-4-1")
ax[0].set_xlabel("Эпоха")
ax[0].set_ylabel("MSE")
ax[0].grid(True)

xt = x_target
ax[1].plot(xt[:n_train], T_tr, "b-", label="Эталон (обучение)")
ax[1].plot(xt[:n_train], Y_tr, "r--", label="ИНС (обучение)")
ax[1].plot(xt[n_train:], T_te, "g-", label="Эталон (тест)")
ax[1].plot(xt[n_train:], Y_te, "m--", label="ИНС (прогноз)")
ax[1].axvline(xt[n_train], color="k", ls=":")
ax[1].set_title("y = 0.2·cos(0.6x) + 0.05·sin(0.6x): эталон и прогноз")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].legend(loc="upper right", fontsize=8)
ax[1].grid(True)

ax[2].plot(xt[:n_train], (T_tr - Y_tr).ravel(), "r", label="Ошибка (обучение)")
ax[2].plot(xt[n_train:], (T_te - Y_te).ravel(), "m", label="Ошибка (тест)")
ax[2].set_title("Ошибка прогноза")
ax[2].set_xlabel("x")
ax[2].set_ylabel("e")
ax[2].legend(fontsize=8)
ax[2].grid(True)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ins_variant6.png")
plt.savefig(out_path, dpi=130)
print(f"\nГрафик сохранён: {out_path}")
plt.show()
