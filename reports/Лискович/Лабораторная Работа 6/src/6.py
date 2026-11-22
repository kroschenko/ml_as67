import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a, b, c, d = 0.3, 0.1, 0.06, 0.1
def target_function(x):
    return a * np.cos(b * x) + c * np.sin(d * x)

x_vals = np.arange(0, 100, 0.5)
y_vals = target_function(x_vals)

N_inputs = 6
H_hidden = 2

X_data, Y_data = [], []
for i in range(len(y_vals) - N_inputs):
    X_data.append(y_vals[i:i+N_inputs])
    Y_data.append(y_vals[i+N_inputs])

X_data = np.array(X_data)
Y_data = np.array(Y_data).reshape(-1, 1)

split_idx = int(len(X_data) * 0.8)
X_train, Y_train = X_data[:split_idx], Y_data[:split_idx]
X_test, Y_test = X_data[split_idx:], Y_data[split_idx:]

y_min, y_max = np.min(Y_train), np.max(Y_train)
X_train_norm = (X_train - y_min) / (y_max - y_min)
Y_train_norm = (Y_train - y_min) / (y_max - y_min)
X_test_norm = (X_test - y_min) / (y_max - y_min)
Y_test_norm = (Y_test - y_min) / (y_max - y_min)

class ElmanRNN:
    def __init__(self, input_size=1, hidden_size=2, output_size=1, learning_rate=0.005):
        np.random.seed(42)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.Wx = np.random.randn(input_size, hidden_size) * 0.1
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Wy = np.random.randn(hidden_size, output_size) * 0.1
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        self.lr = learning_rate
        self.loss_history = []

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, X_seq):
        seq_len = X_seq.shape[0]
        h = np.zeros((1, self.hidden_size))
        h_states = [h]
        for t in range(seq_len):
            x_t = X_seq[t:t+1]
            h = self.sigmoid(np.dot(x_t, self.Wx) + np.dot(h, self.Wh) + self.bh)
            h_states.append(h)
        y = np.dot(h, self.Wy) + self.by
        return y, h_states

    def backward(self, X_seq, Y_true, Y_pred, h_states):
        dY = Y_pred - Y_true
        loss = np.mean(dY**2)
        self.loss_history.append(loss)

        dWy = np.dot(h_states[-1].T, dY)
        dby = dY.copy()

        dh = np.dot(dY, self.Wy.T)

        seq_len = X_seq.shape[0]
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)

        for t in range(seq_len, 0, -1):
            h_t = h_states[t]
            h_prev = h_states[t-1]
            x_t = X_seq[t-1:t]

            dh_raw = dh * h_t * (1 - h_t)

            dWx += np.dot(x_t.T, dh_raw)
            dWh += np.dot(h_prev.T, dh_raw)
            dbh += dh_raw

            dh = np.dot(dh_raw, self.Wh.T)

        self.Wx -= self.lr * dWx
        self.Wh -= self.lr * dWh
        self.Wy -= self.lr * dWy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

    def train(self, X, Y, epochs=500):
        for epoch in range(epochs):
            for i in range(len(X)):
                X_seq = X[i].reshape(-1, 1)
                Y_true = Y[i].reshape(1, 1)
                Y_pred, h_states = self.forward(X_seq)
                self.backward(X_seq, Y_true, Y_pred, h_states)

    def predict(self, X):
        preds = []
        for i in range(len(X)):
            X_seq = X[i].reshape(-1, 1)
            Y_pred, _ = self.forward(X_seq)
            preds.append(Y_pred.flatten()[0])
        return np.array(preds).reshape(-1, 1)

alpha = 0.005
epochs = 500
rnn = ElmanRNN(input_size=1, hidden_size=H_hidden, output_size=1, learning_rate=alpha)
rnn.train(X_train_norm, Y_train_norm, epochs=epochs)

Y_train_pred_norm = rnn.predict(X_train_norm)
Y_train_pred = Y_train_pred_norm * (y_max - y_min) + y_min

Y_test_pred_norm = rnn.predict(X_test_norm)
Y_test_pred = Y_test_pred_norm * (y_max - y_min) + y_min

def create_report_table(y_true, y_pred):
    df = pd.DataFrame({
        'Эталонное значение': y_true.flatten(),
        'Полученное значение': y_pred.flatten()
    })
    df['Отклонение'] = df['Эталонное значение'] - df['Полученное значение']
    return df

print("Результаты обучения")
train_report = create_report_table(Y_train, Y_train_pred)
print(train_report.head(21))

print("\nРезультаты прогнозирования")
test_report = create_report_table(Y_test, Y_test_pred)
print(test_report.head(21))

mse_train = np.mean((Y_train - Y_train_pred)**2)
mse_test = np.mean((Y_test - Y_test_pred)**2)
print(f"\nMSE на обучающей выборке: {mse_train:.6f}")
print(f"MSE на тестовой выборке: {mse_test:.6f}")

plt.figure(figsize=(12, 5))
plt.plot(Y_train, label='Эталонные значения', color='blue', alpha=0.7)
plt.plot(Y_train_pred, label='Прогноз РНС (Элмана)', color='red', linestyle='dashed')
plt.title('График прогнозируемой функции на участке обучения (РНС Элмана)')
plt.xlabel('Шаг времени')
plt.ylabel('Значение y')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(rnn.loss_history, color='green')
plt.title(f'График изменения ошибки (MSE) от итерации. Alpha = {alpha}')
plt.xlabel('Итерация')
plt.ylabel('Ошибка (Loss)')
plt.grid(True)
plt.show()