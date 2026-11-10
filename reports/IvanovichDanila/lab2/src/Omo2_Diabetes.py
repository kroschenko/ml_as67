import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Настройки для графиков
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    print("--- ЗАДАЧА 2: КЛАССИФИКАЦИЯ (Pima Indians Diabetes) ---")

    # 1. Загрузка данных
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    try:
        df = pd.read_csv('pima-indians-diabetes.csv', comment='#', header=None, names=col_names)
    except FileNotFoundError:
        print("Ошибка: Файл 'pima-indians-diabetes.csv' не найден в текущей директории.")
        return

    # Разделение на признаки и целевую переменную
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 2. Стандартизация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 3. Обучение модели логистической регрессии
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)

    # Предсказание
    y_pred = log_reg.predict(X_test)

    # 4. Расчет метрик
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy (Общая точность): {accuracy:.4f}")
    print(f"Precision (Точность): {precision:.4f}")
    print(f"Recall (Полнота): {recall:.4f}")

    # 5. Матрица ошибок и выводы
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Здоров (0)', 'Диабет (1)'], 
                yticklabels=['Здоров (0)', 'Диабет (1)'])
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.tight_layout()
    plt.show()

    # Анализ матрицы ошибок
    tn, fp, fn, tp = cm.ravel()
    print("\n--- Выводы по матрице ошибок ---")
    print(f"True Positives (TP - верно предсказан диабет): {tp}")
    print(f"True Negatives (TN - верно предсказано здоровье): {tn}")
    print(f"False Positives (FP - здорового назвали больным): {fp}")
    print(f"False Negatives (FN - больного назвали здоровым): {fn}")

    print("\nАнализ:")
    print("В медицинских задачах ложноотрицательные срабатывания (FN) считаются наиболее опасными,")
    print("так как они означают, что модель пропустила болезнь у пациента.")
    print(f"В нашей модели FN = {fn}. Это означает, что {fn} больных людей были ошибочно признаны здоровыми.")

if __name__ == "__main__":
    main()