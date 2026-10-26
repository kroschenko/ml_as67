import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as mplt
from scipy import stats
import numpy as np
df = pd.read_csv("student-por.csv", sep = ';')

X = df[['studytime', 'failures', 'G1', 'G2']]
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

r2 = model.score(X_test, y_test)
print('R^2 на тесте:', r2)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print('Предсказания:', y_pred[:10])
print('MAE:', mae)

x_p = df['G2']
y_p = df['G3']

slope, intercept, rvalue, pvalue, std_err = stats.linregress(df['G2'], df['G3'])

x_line = np.linspace(df['G2'].min(), df['G2'].max(), 100)
y_line = intercept + slope * x_line

mplt.figure(figsize=(10, 6))
mplt.scatter(x_p, y_p, alpha=0.7, color='steelblue', label='Студенты')

mplt.plot(x_line, y_line, color='red', linewidth=2)

mplt.xlabel('G2 - оценка за предыдущий период')
mplt.ylabel('G3 - итоговая оценка')
mplt.title('Зависимость G3 от G2')
mplt.legend()
mplt.grid(alpha=0.3)

mplt.show()