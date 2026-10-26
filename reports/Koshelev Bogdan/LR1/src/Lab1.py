import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)

df = pd.read_csv(r'heart.csv')

print("1. ИНФОРМАЦИЯ О ДАТАСЕТЕ")
print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())
print("\nИнформация о датафрейме:")
df.info()
print("\nПропуски по столбцам:")
print(df.isnull().sum())
print("Всего пропусков:", df.isnull().sum().sum())


counts = df['target'].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(['Здоровые (0)', 'Больные (1)'], counts.values, color=['#2ecc71', '#e74c3c'])
plt.title('Количество здоровых и больных пациентов')
plt.xlabel('Состояние')
plt.ylabel('Количество пациентов')
for i, v in enumerate(counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontsize=11)
plt.savefig('graph1_target.png', dpi=100)
plt.show()

print("\nЗдоровых:", counts[0], "| Больных:", counts[1])


plt.figure(figsize=(9, 6))
plt.scatter(df[df.target == 0]['age'], df[df.target == 0]['thalach'],
            c='#2ecc71', label='Здоровые', alpha=0.7, edgecolors='k')
plt.scatter(df[df.target == 1]['age'], df[df.target == 1]['thalach'],
            c='#e74c3c', label='Больные', alpha=0.7, edgecolors='k')
plt.title('Зависимость максимального пульса от возраста')
plt.xlabel('Возраст (age)')
plt.ylabel('Максимальный пульс (thalach)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graph2_scatter.png', dpi=100)
plt.show()

df['sex'] = df['sex'].map({0: 'female', 1: 'male'})
print("\nУникальные значения sex:", df['sex'].unique())

df = pd.get_dummies(df, columns=['sex'], drop_first=True)
print("Столбцы после One-Hot Encoding:")
print(df.columns.tolist())

mean_chol = df.groupby('target')['chol'].mean()
print("\nСредний холестерин:")
print(f"  Здоровые (0): {mean_chol[0]:.2f}")
print(f"  Больные   (1): {mean_chol[1]:.2f}")

plt.figure(figsize=(6, 4))
plt.bar(['Здоровые', 'Больные'], [mean_chol[0], mean_chol[1]],
        color=['#2ecc71', '#e74c3c'])
plt.title('Средний уровень холестерина')
plt.ylabel('chol')
plt.savefig('graph3_chol.png', dpi=100)
plt.show()


cols = ['age', 'trestbps', 'chol', 'thalach']

print("\nДО нормализации:")
print(df[cols].describe().loc[['min', 'max', 'mean']])

for col in cols:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

print("\nПОСЛЕ нормализации:")
print(df[cols].describe().loc[['min', 'max', 'mean']])

print("\nГотово. Графики сохранены в файлы graph1_target.png, graph2_scatter.png, graph3_chol.png")