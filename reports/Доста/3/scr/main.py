import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("breast_cancer.csv")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

X = df.drop(columns=["id", "diagnosis"])
y = df["diagnosis"].map({"B": 0, "M": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_k, best_score = 1, 0
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5).mean()
    if score > best_score:
        best_k, best_score = k, score

print(f"Лучшее k = {best_k} (CV accuracy = {best_score:.4f})\n")

models = {
    f"k-NN (k={best_k})": KNeighborsClassifier(n_neighbors=best_k),
    "Decision Tree":      DecisionTreeClassifier(),
    "SVM":                SVC(kernel="rbf"),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Реально benign", "Реально malignant"],
        columns=["Предсказано benign", "Предсказано malignant"]
    )
    tn, fp, fn, tp = cm.ravel()

    print(f"Модель: {name}")
    print("Матрица ошибок:")
    print(cm_df)
    print(classification_report(
        y_test, y_pred,
        target_names=["benign", "malignant"],
        digits=4
    ))
    print(f"Ложноотрицательных прогнозов (M→B): {fn}")