import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1 Загрузка и статистика
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv('pima-indians-diabetes.csv', header=None,
                 names=cols, comment='#')

print("Размер:", df.shape)
print(df.dtypes)
print("\nПропуски (NaN):", df.isnull().sum().sum())
print("\nСтатистика:")
print(df.describe().loc[['mean', '50%', 'std']])

# 2 Замена нулей медианой
for col in ['Glucose', 'BloodPressure', 'SkinThickness']:
    med = df.loc[df[col] != 0, col].median()
    df.loc[df[col] == 0, col] = med
print("\nНули заменены медианой в Glucose, BloodPressure, SkinThickness")

# 3 Гистограммы BMI и AGE
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(df['BMI'], bins=25, color='skyblue', edgecolor='black')
ax[0].set_title('BMI')
ax[1].hist(df['Age'], bins=25, color='orange', edgecolor='black')
ax[1].set_title('Age')
plt.tight_layout()
plt.savefig('graph1_hist.png', dpi=100)
plt.show()

# 4 Матрица корреляции
corr = df[['Glucose', 'BMI', 'Age', 'Outcome']].corr()
print("\nКорреляция:\n", corr.round(3))

plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(4), corr.columns, rotation=45)
plt.yticks(range(4), corr.columns)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center')
plt.title('Матрица корреляции')
plt.tight_layout()
plt.savefig('graph2_corr.png', dpi=100)
plt.show()

# 5 Круговая диаграмма Outcome
counts = df['Outcome'].value_counts()
print("\nOutcome:", dict(counts))

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=['Без диабета', 'Диабет'],
        colors=['limegreen', 'tomato'],
        autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05))
plt.title('Распределение Outcome')
plt.savefig('graph3_pie.png', dpi=100)
plt.show()

# 6 Стандартизация (все признаки кроме Outcome)
features = [c for c in df.columns if c != 'Outcome']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
print("\nПосле стандартизации (mean~0, std~1):")
print(df[features].describe().loc[['mean', 'std']].round(3))

# 7 Диаграмма рассеяния (Glucose vs BMI по классам)
for val, color, label in [(0, 'limegreen', 'Без диабета'), (1, 'tomato', 'Диабет')]:
    sub = df[df['Outcome'] == val]
    plt.scatter(sub['Glucose'], sub['BMI'], c=color, label=label,
                alpha=0.7, edgecolors='k')
plt.xlabel('Glucose (std)')
plt.ylabel('BMI (std)')
plt.title('Glucose vs BMI')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graph4_scatter.png', dpi=100)
plt.show()