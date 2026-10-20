import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# data = datasets.load_digits(as_frame=True)
# digits_df = pd.DataFrame(data=data['data'], columns=data['feature_names'])
# digits_df['target'] = data['target']
# digits_df.to_csv('digits.csv', index=False)


def solution() -> None:
    TARGET : str = 'target'
    digits : pd.DataFrame = pd.read_csv('digits.csv', sep=',')
    
    # Обучение моделей многоклассовой классификации
    X_train, X_test, y_train, y_test = train_test_split(
                digits.drop(columns=TARGET),
                digits[TARGET],
                test_size=0.25,
                random_state=42
        )
    knn_model = KNeighborsClassifier()
    dt_model = DecisionTreeClassifier()
    svc_model = SVC()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_model.fit(X_train_scaled, y_train)
    knn_y_pred = knn_model.predict(X_test_scaled)
    
    dt_model.fit(X_train_scaled, y_train)
    dt_y_pred = dt_model.predict(X_test_scaled)

    svc_model.fit(X_train_scaled, y_train)
    svc_y_pred = knn_model.predict(X_test_scaled)

    # Оценка качества обученных моделей
    knn_report = classification_report(y_test, knn_y_pred)
    dt_report = classification_report(y_test, dt_y_pred)
    svc_report = classification_report(y_test, svc_y_pred)
    print("Оценка модели kNN:\n", knn_report)
    print("Оценка модели DT:\n", dt_report)
    print("Оценка модели SVC:\n", svc_report)

if __name__ == "__main__":
    solution()