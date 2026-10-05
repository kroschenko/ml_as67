import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print('Задание 1')
df = pd.read_csv('adult.csv',sep=',')
print(df.head(10))

print('Задание 2')
high = df['workclass'].value_counts().index[0]

#df['workclass'].isin(['?'])

df['workclass'] = df['workclass'].replace('?', high)
print(df.head(10))

print('Задание 3')
sex_counts = df['sex'].value_counts()
print(sex_counts)

plt.figure(figsize=(6, 5))
plt.bar(sex_counts.index, sex_counts.values, color=['blue', 'pink'], edgecolor='black')
plt.title('Распределение по полу')
plt.xlabel('Пол')
plt.ylabel('Количество')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print('Задание 4')
df = pd.get_dummies(df, columns =['race'], prefix='race', drop_first=True)
print(df.columns.tolist())

print('Задание 5')
df['income'] = df['income'].str.replace('.', '', regex=False)

low  = df[df['income'] == '<=50K']['age']
high = df[df['income'] == '>50K']['age']

age_min = df['age'].min()
age_max = df['age'].max()
step = 2

bins = np.arange(age_min, age_max + step, step)

plt.figure(figsize=(10, 6))
plt.hist(low, bins = bins, alpha=0.5, label='<=50K', color='steelblue', edgecolor='black')
plt.hist(high,bins=bins, alpha=0.5, label='>50K',  color='tomato',    edgecolor='black')

plt.title('Распределение возраста по уровню дохода')
plt.xlabel('Age')
plt.ylabel('Количество человек')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print('Задание 6')
df['is_usa'] = np.where(df['native.country'] == 'United-States', 1, 0)
print(df.head(10))

