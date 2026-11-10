import torch
from torch import nn
from torch import optim

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Платформа обучения: ", device)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 16) -> None:
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=1),
        )

    def forward(self, x):
        return self.linear_stack(x)


def train_and_evaluate(hidden: int, data, epochs: int = 200) -> dict:
    X_train_t, X_test_t, y_train_t, y_test_t = data

    torch.manual_seed(42)
    model = MLP(in_features=X_train_t.shape[1], hidden=hidden).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nСкрытый слой: {hidden} нейронов")
    print("\nОбучаемая модель: \n", model)
    print("\tКоличество эпох: ", epochs)
    print("\tОптимизатор(Adam): ")
    for name, val in optimizer.defaults.items():
        print("\t\t", name, " : ", val)
    print("\tФункция потерь(бинарная): ", criterion)
    print("=" * 45)

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Эпоха {epoch + 1}/{epochs}, loss = {loss.item():.4f}")

    print("\nОценка модели нейронной сети: ")
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        predicted_classes = (logits > 0).float()

    y_true = y_test_t.cpu().numpy().ravel()
    y_hat = predicted_classes.cpu().numpy().ravel()

    print(classification_report(
        y_true, y_hat,
        target_names=['доброкачественная (B)', 'злокачественная (M)'],
        zero_division=0,
    ))

    return dict(
        acc=accuracy_score(y_true, y_hat),
        prec=precision_score(y_true, y_hat, zero_division=0),
        rec=recall_score(y_true, y_hat, zero_division=0),
        f1=f1_score(y_true, y_hat, zero_division=0),
    )


def solution() -> None:
    df = pd.read_csv('breast_cancer.csv')
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')].drop(columns='id')
    y = (df['diagnosis'] == 'M').astype(int)
    X = df.drop(columns='diagnosis').astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    data = (X_train_t, X_test_t, y_train_t, y_test_t)

    results = {}
    for hidden in [16, 32]:
        results[hidden] = train_and_evaluate(hidden, data)

    print("\n" + "=" * 45)
    print("Сравнение метрик (16 vs 32 нейрона):")
    print(f"\t{'hidden':>8s} | {'accuracy':>8s} | {'precision':>9s} | {'recall':>6s} | {'f1':>6s}")
    for hidden, r in results.items():
        print(f"\t{hidden:>8d} | {r['acc']:>8.4f} | {r['prec']:>9.4f} | "
              f"{r['rec']:>6.4f} | {r['f1']:>6.4f}")

    print("\nВывод по эксперименту:")
    print(f"\tprecision: 16 -> {results[16]['prec']:.4f}, 32 -> {results[32]['prec']:.4f}")
    print(f"\trecall   : 16 -> {results[16]['rec']:.4f}, 32 -> {results[32]['rec']:.4f}")

if __name__ == "__main__":
    solution()