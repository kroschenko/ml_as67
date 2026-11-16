
import os
import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score
# Выбор устройства
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Платформа обучения:", device)
TARGET = "Outcome"
COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    TARGET
]
EPOCHS = 500
# Класс многослойного перцептрона
class MLP(nn.Module):
    def __init__(self, in_features, two_hidden=True):
        super().__init__()

        if two_hidden:
            self.layers = nn.Sequential(
                nn.Linear(in_features, 12),
                nn.ReLU(),

                nn.Linear(12, 8),
                nn.ReLU(),

                nn.Linear(8, 1)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_features, 12),
                nn.ReLU(),

                nn.Linear(12, 1)
            )
    def forward(self, x):
        return self.layers(x)
# Загрузка и подготовка данных
def load_data():
    # Путь к файлу
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        base_dir,
        "pima-indians-diabetes.csv"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Файл не найден: {file_path}"
        )
    # Загрузка датасета
    df = pd.read_csv(
        file_path,
        header=None,
        names=COLUMNS,
        comment="#",
        sep=","
    )
    # Преобразование всех столбцов в числа
    for column in COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )
    # Удаление строк без целевого значения
    df = df.dropna(subset=[TARGET])
    # Оставляем только классы 0 и 1
    df = df[df[TARGET].isin([0, 1])]
    # Удаление полностью пустых строк
    df = df.dropna(how="all")
    print("\nРазмер датасета:", df.shape)
    # Разделение признаков и целевой переменной
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(np.float32)
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
    # Замена нулевых значений на пропуски
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]
    X_train[zero_columns] = X_train[zero_columns].replace(
        0, np.nan
    )
    X_test[zero_columns] = X_test[zero_columns].replace(
        0, np.nan
    )
    # Заполнение пропусков медианами
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    # Стндартизация признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Преобразование в тензоры PyTorch
    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    ).to(device)
    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(device)
    y_train = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32
    ).unsqueeze(1).to(device)
    y_test = y_test.to_numpy().astype(int)
    return X_train, X_test, y_train, y_test
# Обучение и оценка модели
def train_and_evaluate(two_hidden, data):
    X_train, X_test, y_train, y_test = data
    torch.manual_seed(42)
    model = MLP(
        in_features=X_train.shape[1],
        two_hidden=two_hidden
    ).to(device)
    # Функция потерь
    criterion = nn.BCEWithLogitsLoss()
    # Оптимизатор
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )
    print("\nАрхитектура модели:")
    print(model)
    print("\nКоличество эпох:", EPOCHS)
    print("Оптимизатор: Adam")
    print("Функция потерь: BCEWithLogitsLoss")
    # Цикл обучения
    for epoch in range(EPOCHS):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(
                f"Эпоха {epoch + 1}/{EPOCHS}, "
                f"ошибка: {loss.item():.4f}"
            )
    # Оценка модели
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probabilities = torch.sigmoid(logits)
        predictions = (
            probabilities >= 0.5
        ).int().squeeze(1).cpu().numpy()
    print("\nОтчет классификации:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "Нет диабета (0)",
                "Диабет (1)"
            ],
            zero_division=0
        )
    )
    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )
    print(
        f"Recall для класса диабет: {recall:.4f}"
    )
    return recall
# Основная функция
def solution():
    data = load_data()
    # Первая модель
    print("\n" + "=" * 50)
    print("МОДЕЛЬ 1: ДВА СКРЫТЫХ СЛОЯ (12 И 8)")
    print("=" * 50)
    recall_two = train_and_evaluate(
        two_hidden=True,
        data=data
    )
    # Вторая модель
    print("\n" + "=" * 50)
    print("МОДЕЛЬ 2: ОДИН СКРЫТЫЙ СЛОЙ (12)")
    print("=" * 50)
    recall_one = train_and_evaluate(
        two_hidden=False,
        data=data
    )
    # Сравнение моделей
    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 50)
    print(
        f"Два скрытых слоя (12, 8): {recall_two:.4f}"
    )
    print(
        f"Один скрытый слой (12): {recall_one:.4f}"
    )
    if recall_two > recall_one:
        print(
            "\nПредпочтительнее модель с двумя скрытыми слоями."
        )
    elif recall_one > recall_two:
        print(
            "\nПредпочтительнее модель с одним скрытым слоем."
        )
    else:
        print("\nОбе модели показали одинаковый Recall.")
    print("=" * 50)
if __name__ == "__main__":
    solution()
