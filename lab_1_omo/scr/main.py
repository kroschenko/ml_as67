import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

def show_menu():
    print("""
╔══════════════════════════════════════════════════════════╗
║                    МЕНЮ ЗАДАНИЙ                          ║
╠══════════════════════════════════════════════════════════╣
║  1 — Информация о заемшиках                              ║
║  2 — Топ-5 целей кредита (Purpose)                       ║
║  3 — Кодирование Sex и Housing                           ║
║  4 — Boxplot: Credit amount (хорошие vs плохие)          ║
║  5 — Сводная: возраст и длительность по credit_history   ║
║  6 — Нормализация Age, Credit amount, Duration           ║
║  0 — Выход                                               ║
╚══════════════════════════════════════════════════════════╝
""")


def level1(gerCred):
    print("\n" + "=" * 60)
    print("             ИНФОРМАЦИЯ О ЗАЁМЩИКАХ")
    print("=" * 60)

    total = len(gerCred)
    good = (gerCred['default'] == 0).sum()
    bad = (gerCred['default'] == 1).sum()

    print("\n=== ОБЩАЯ ИНФОРМАЦИЯ ===")
    print(f"Всего заёмщиков:       {total}")
    print(f"Хороших заёмщиков:     {good} ({good / total * 100:.1f}%)")
    print(f"Плохих заёмщиков:      {bad} ({bad / total * 100:.1f}%)")

    print("\n=== ВОЗРАСТ ЗАЁМЩИКОВ ===")
    print(f"Средний возраст:       {gerCred['age'].mean():.1f} лет")
    print(f"Минимальный возраст:   {gerCred['age'].min()} лет")
    print(f"Максимальный возраст:  {gerCred['age'].max()} лет")

    print("\n=== СУММЫ КРЕДИТОВ ===")
    print(f"Средняя сумма:         {gerCred['credit_amount'].mean():,.2f} DM")
    print(f"Минимальная сумма:     {gerCred['credit_amount'].min():,} DM")
    print(f"Максимальная сумма:    {gerCred['credit_amount'].max():,} DM")

    print("\n=== ДЛИТЕЛЬНОСТЬ КРЕДИТОВ ===")
    print(f"Средний срок:          {gerCred['duration_in_month'].mean():.1f} месяцев")
    print(f"Минимальный срок:      {gerCred['duration_in_month'].min()} месяцев")
    print(f"Максимальный срок:     {gerCred['duration_in_month'].max()} месяцев")

    print("\n=== ЖИЛИЩНЫЕ УСЛОВИЯ ===")
    for name, count in gerCred['housing'].value_counts().items():
        percent = count / total * 100
        print(f"{name:<15} {count:>5} чел. ({percent:.1f}%)")

    print("\n=== СЕМЕЙНЫЙ СТАТУС И ПОЛ ===")
    for name, count in gerCred['personal_status_sex'].value_counts().items():
        print(f"{name:<45} {count:>5} чел.")

    print("\n=== КРЕДИТНАЯ ИСТОРИЯ ===")
    for name, count in gerCred['credit_history'].value_counts().items():
        print(f"{name[:45]:<45} {count:>5} чел.")

    print("\n" + "=" * 60)
    print("           ПРИМЕРЫ ЗАЁМЩИКОВ")
    print("=" * 60)

    for number, (_, row) in enumerate(gerCred.head(5).iterrows(), start=1):
        status = "Хороший" if row['default'] == 0 else "Плохой"

        print(f"\nЗаёмщик №{number}")
        print("-" * 40)
        print(f"Возраст:           {row['age']} лет")
        print(f"Сумма кредита:     {row['credit_amount']:,} DM")
        print(f"Срок кредита:      {row['duration_in_month']} месяцев")
        print(f"Жильё:             {row['housing']}")
        print(f"Семейный статус:   {row['personal_status_sex']}")
        print(f"Кредитная история: {row['credit_history']}")
        print(f"Результат:         {status} заёмщик")

    print("\n" + "=" * 60)



def level2 (gerCred):
    print(gerCred['purpose'].value_counts().head(5))
    



def level3 (gerCred):
    # unique метод вывода уникального значения только 1 раз 
    print(gerCred['personal_status_sex'].unique()) 
    print(gerCred['housing'].unique()) 
    print("=== ДО ===")
    print(gerCred[['personal_status_sex', 'housing']].head(10)) 
    #метод head являеться методов отображения конкретных строк
  
    pss_map = {
        'male : single'                          : 0,
        'male : married/widowed'                 : 1,
        'male : divorced/separated'              : 2,
        'female : divorced/separated/married'    : 3,
        'female : single'                        : 4,
    }

    housing_map = {
        'own'      : 2,
        'rent'     : 1,
        'for free' : 0,   
    }
    print("Sex:", pss_map)
    print("Housing:", housing_map)
    gerCred['personal_status_sex'] = gerCred['personal_status_sex'].map(pss_map)
    gerCred['housing'] = gerCred['housing'].map(housing_map)
    print(gerCred[['personal_status_sex', 'housing']].head(10)) 

def level4 (gerCred):

    good = gerCred.loc[gerCred['default'] == 0, 'credit_amount'] # loc = условие/метка строки
    bad  = gerCred.loc[gerCred['default'] == 1, 'credit_amount']
    print("Хороших:", len(good))
    print("Плохих: ", len(bad))
    print("Всего:  ", len(good) + len(bad))   

    # ─── Boxplot ───
    plt.figure(figsize=(8, 6))
    plt.boxplot([good, bad], tick_labels=['Хорошие', 'Плохие'],
                flierprops=dict(marker='.', markersize=4, alpha=0.4)  )
                # маркер отображения----^ , размерность--^ , ^-прозрачность 
    plt.axhline(y=0)
    plt.axhspan( 2400, 2390, color= 'red' )  
 #                координаты-^ , ^-размерность     
    plt.yscale('log') # log метод делает разницу более наглядной
    plt.ylabel('Сумма кредита (log)') 
    plt.title('Сумма кредита: хорошие vs плохие (log-шкала)')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def level5 (gerCred):
    result = gerCred.groupby('credit_history').agg(
        средний_возраст=('age', 'mean'),
        средняя_длительность=('duration_in_month', 'mean')
    )
    print(result)
        
def level6(gerCred):
    cols = ['age', 'credit_amount', 'duration_in_month']

    print("\n" + "=" * 75)
    print("                  НОРМАЛИЗАЦИЯ ДАННЫХ")
    print("=" * 75)

    for col in cols:
        min_val = gerCred[col].min()
        max_val = gerCred[col].max()

        # Нормализация Min-Max в диапазон от 0 до 1
        if max_val != min_val:
            gerCred[col + '_norm'] = (
                (gerCred[col] - min_val) / (max_val - min_val)
            )
        else:
            gerCred[col + '_norm'] = 0.0

        print(f"\n=== {col.upper()} ===")
        print(f"Минимальное значение: {min_val}")
        print(f"Максимальное значение: {max_val}")

    print("\n" + "=" * 75)
    print("             РЕЗУЛЬТАТЫ НОРМАЛИЗАЦИИ")
    print("=" * 75)

    result = gerCred[
        ['age', 'age_norm',
         'credit_amount', 'credit_amount_norm',
         'duration_in_month', 'duration_in_month_norm']
    ].head(10).copy()

    result.columns = [
        'Возраст', 'Возраст_norm',
        'Сумма кредита', 'Сумма_norm',
        'Срок кредита', 'Срок_norm'
    ]

    print(result.round(3).to_string(index=False))

    print("\n" + "=" * 75)
    print("             СТАТИСТИКА ПОСЛЕ НОРМАЛИЗАЦИИ")
    print("=" * 75)

    norm_cols = ['age_norm', 'credit_amount_norm', 'duration_in_month_norm']
    print(gerCred[norm_cols].describe().round(3))

    print("\nНормализация завершена!")
    print("Новые столбцы добавлены в датасет.")

if __name__ == "__main__":
    gerCred = pd.read_csv("german_credit.csv") # подключение к файлу
    while True:
        show_menu()
        choice = input("Выбери пункт: ").strip()
        match choice:
            case '0':
                print("Выход. Пока пока!")
                break
            case '1':
                level1(gerCred)
            case '2':
                level2(gerCred)
            case '3':
                level3(gerCred)
            case '4':
                level4(gerCred)
            case '5':
                level5(gerCred)
            case '6':
                level6(gerCred)
            case _:
                print("Попробуй снова.\n")
                continue

        input("\nНажми Enter, чтобы вернуться в меню...")
        print()
    