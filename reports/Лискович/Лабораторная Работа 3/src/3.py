import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

df = pd.read_csv('Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df = df.drop('customerID', axis=1)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier()
dt = DecisionTreeClassifier(random_state=42)
svm = SVC(random_state=42)

knn.fit(X_train_scaled, y_train)
dt.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
y_pred_dt = dt.predict(X_test_scaled)
y_pred_svm = svm.predict(X_test_scaled)

f1_knn = f1_score(y_test, y_pred_knn)
f1_dt = f1_score(y_test, y_pred_dt)
f1_svm = f1_score(y_test, y_pred_svm)

print(f"F1-score k-NN: {f1_knn:.4f}")
print(f"F1-score Decision Tree: {f1_dt:.4f}")
print(f"F1-score SVM: {f1_svm:.4f}")

best_model = max(('k-NN', f1_knn), ('Decision Tree', f1_dt), ('SVM', f1_svm), key=lambda x: x[1])[0]

print(f"Самый эффективный метод на данном датасете - {best_model}")