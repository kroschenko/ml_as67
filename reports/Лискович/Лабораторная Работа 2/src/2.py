import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score, 
                             precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

def regression_task():
    try:
        df = pd.read_csv('CarPrice_Assignment.csv')
    except FileNotFoundError:
        print("Ошибка: Файл 'CarPrice_Assignment.csv' не найден.")
        return

    features = ['horsepower', 'citympg', 'enginesize', 'curbweight', 'highwaympg']
    target = 'price'
    
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"Внимание: В датасете отсутствуют колонки {missing_cols}.")
        return

    df_reg = df[features + [target]].dropna()
    
    X = df_reg[features]
    y = df_reg[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("Задача Регрессии (Предсказание цены)")
    print(f"R2 (коэффициент детерминации): {r2:.4f}")
    print(f"MAE (средняя абсолютная ошибка): {mae:.4f}\n")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_reg['horsepower'], df_reg['price'], alpha=0.5, color='blue', label='Данные')
    
    z = np.polyfit(df_reg['horsepower'], df_reg['price'], 1)
    p = np.poly1d(z)
    plt.plot(df_reg['horsepower'], p(df_reg['horsepower']), color='red', linewidth=2, label='Линия тренда (по horsepower)')
    
    plt.title('Зависимость цены автомобиля от мощности')
    plt.xlabel('Horsepower')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def classification_task():
    try:
        df = pd.read_csv('adult.csv')
    except FileNotFoundError:
        print("Ошибка: Файл 'adult.csv' не найден.")
        return

    df = df.replace(' ?', np.nan)
    df = df.dropna()
    
    target_col = 'income' if 'income' in df.columns else 'salary'
    if target_col in df.columns:
        df[target_col] = df[target_col].str.strip()
        df['target'] = df[target_col].apply(lambda x: 1 if x in ['>50K', '>50K.'] else 0)
    else:
        print("Ошибка: Не найдена целевая колонка 'income' или 'salary'.")
        return

    X = df.drop([target_col, 'target'], axis=1)
    y = df['target']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = pd.get_dummies(X_train_raw, drop_first=True)
    X_test = pd.get_dummies(X_test_raw, drop_first=True)
    
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] # Вероятности для AUC
    
    # Считаем метрики со взвешенным средним (как в таблице)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print("Задача Классификации (Доход > 50K)")
    print(f"AUC:     {auc:.3f}")
    print(f"CA:      {accuracy:.3f}")
    print(f"F1:      {f1:.3f}")
    print(f"Prec:    {precision:.3f}")
    print(f"Recall:  {recall:.3f}")
    print(f"MCC:     {mcc:.3f}\n")
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])
    
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.show()

if __name__ == "__main__":
    regression_task()
    classification_task()