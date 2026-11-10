import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score

device = (
    torch.accelerator.current_accelerator().type  # type: ignore
    if torch.accelerator.is_available()
    else "cpu"
)
print("Платформа обучения: ", device)

TARGET: str = "Outcome"  # 1 - диабет, 0 - нет диабета
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", TARGET,
]
EPOCHS: int = 500


class MLP(nn.Module):
    """
    two_hidden=True : 8 -> 12 (ReLU) -> 8 (ReLU) -> 1
    two_hidden=False: 8 -> 12 (ReLU) -> 1   (эксперимент: второй слой удалён)
    """

    def __init__(self, in_features: int, two_hidden: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Linear(in_features=in_features, out_features=12),
            nn.ReLU(),
        ]
        if two_hidden:
            layers += [
                nn.Linear(in_features=12, out_features=8),
                nn.ReLU(),
                nn.Linear(in_features=8, out_features=1),
            ]
        else:
            layers += [nn.Linear(in_features=12, out_features=1)]
        # Sigmoid в модель не добавляем: он встроен в nn.BCEWithLogitsLoss,
        # а при предсказании применяется явно (torch.sigmoid / порог logits > 0)
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_relu_stack(x)


def load_data():
    df: pd.DataFrame = pd.read_csv(
        "pima-indians-diabetes.csv", sep=",", comment="#", header=None, names=COLUMNS
    )

    # В этих столбцах 0 означает пропуск -> заменяем на медиану
    zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_nan] = df[zero_as_nan].replace(0, np.nan)

    X = df.drop(columns=TARGET)
    y = df[TARGET].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Медианы считаем только по обучающей выборке (без утечки данных)
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    # для BCEWithLogitsLoss метки: float32, форма (N, 1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_np = y_test.astype(int)

    return X_train_t, X_test_t, y_train_t, y_test_np


def train_and_evaluate(two_hidden: bool, data) -> float:
    X_train, X_test, y_train, y_test = data

    torch.manual_seed(42)
    model = MLP(in_features=X_train.shape[1], two_hidden=two_hidden).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("=" * 45)
    print("\nОбучаемая модель: \n", model)
    print("=" * 45)
    print("\nЦикл обучения модели: ")
    print("\tКоличество эпох: ", EPOCHS)
    print("\tОптимизатор(Adam): ")
    for name, val in optimizer.defaults.items():
        print("\t\t", name, " : ", val)
    print("\tФункция потерь(бинарная): ", criterion)
    print("=" * 45)

    # Цикл обучения
    for _ in range(EPOCHS):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Оценка модели
    print("\nОценка модели нейронной сети: ")
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits)  # Sigmoid на выходе
        predicted = (probs > 0.5).int().squeeze(1).cpu().numpy()

    print(classification_report(
        y_test, predicted, target_names=["нет диабета (0)", "диабет (1)"]
    ))
    return recall_score(y_test, predicted, pos_label=1)


def solution() -> None:
    data = load_data()

    print("\n##### МОДЕЛЬ 1: два скрытых слоя (12 и 8) #####")
    recall_two = train_and_evaluate(two_hidden=True, data=data)

    print("\n##### МОДЕЛЬ 2: один скрытый слой (12) #####")
    recall_one = train_and_evaluate(two_hidden=False, data=data)

    print("=" * 45)
    print("Сравнение recall для класса 'диабет':")
    print(f"\tДва скрытых слоя (12, 8): {recall_two:.3f}")
    print(f"\tОдин скрытый слой (12):   {recall_one:.3f}")
    print("=" * 45)


if __name__ == "__main__":
    solution()
