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

y_min, y_max = np.min(y_vals), np.max(y_vals)
X_train_norm = (X_train - y_min) / (y_max - y_min)
Y_train_norm = (Y_train - y_min) / (y_max - y_min)
X_test_norm = (X_test - y_min) / (y_max - y_min)
Y_test_norm = (Y_test - y_min) / (y_max - y_min)

class SimpleNN:
    def __init__(self, inputs, hidden, outputs=1, learning_rate=0.01):
        np.random.seed(42)
        self.W1 = np.random.randn(inputs, hidden) * 0.1
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, outputs) * 0.1
        self.b2 = np.zeros((1, outputs))
        
        self.lr = learning_rate
        self.loss_history = []

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

    def backward(self, X, Y_true, Y_pred):
        error = Y_pred - Y_true
        loss = np.mean(error**2)
        self.loss_history.append(loss)

        dZ2 = error
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, Y, epochs=2000):
        for epoch in range(epochs):
            Y_pred = self.forward(X)
            self.backward(X, Y, Y_pred)

alpha = 0.01
epochs = 2000
nn = SimpleNN(inputs=N_inputs, hidden=H_hidden, learning_rate=alpha)
nn.train(X_train_norm, Y_train_norm, epochs=epochs)

Y_train_pred_norm = nn.forward(X_train_norm)
Y_train_pred = Y_train_pred_norm * (y_max - y_min) + y_min

Y_test_pred_norm = nn.forward(X_test_norm)
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

plt.figure(figsize=(12, 5))
plt.plot(Y_train, label='Эталонные значения', color='blue', alpha=0.7)
plt.plot(Y_train_pred, label='Прогноз ИНС', color='red', linestyle='dashed')
plt.title('График прогнозируемой функции на участке обучения')
plt.xlabel('Шаг времени')
plt.ylabel('Значение y')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(nn.loss_history, color='green')
plt.title(f'График изменения ошибки (MSE) от итерации. Alpha = {alpha}')
plt.xlabel('Итерация (Эпоха)')
plt.ylabel('Ошибка (Loss)')
plt.grid(True)
plt.show()