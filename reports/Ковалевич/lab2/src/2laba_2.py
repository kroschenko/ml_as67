import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

df = pd.read_csv('mushrooms.csv', sep = ',')

X = df.drop('class', axis = 'columns')
y = df['class']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='p')
recall = recall_score(y_test, y_pred, pos_label = 'p')

print('accuracy =' + str(accuracy))
print('precision =' + str(precision))
print('recall =' + str(recall))

cm = confusion_matrix(y_test, y_pred)
print('Вывод матрицы ошибок в обычном виде')
print(cm)

print('Вывод матрицы ошибок в виде таблицы')

tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
table = [
    ['Реально e', 'TN = ' + str(tn), 'FP = ' + str(fp)],
    ['Реально p', 'FN = ' + str(fn), 'TP = ' + str(tp)],
]
print(tabulate(table,
               headers=['', 'Предсказано e', 'Предсказано p'],
               tablefmt='grid'))