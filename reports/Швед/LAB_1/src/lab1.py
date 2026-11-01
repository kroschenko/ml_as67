import pandas as pd
import matplotlib.pyplot as plt


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 100)
pd.set_option("display.max_colwidth", 25)


def main():
    # 1. Загрузка данных, первые 5 строк и информация о столбцах
    df = pd.read_csv("Titanic-Dataset.csv")

    print(df.head())
    df.info()

    # 2. Количество выживших и погибших + столбчатая диаграмма
    survived_counts = df["Survived"].value_counts().sort_index()
    print("\nКоличество погибших (0) и выживших (1):")
    print(survived_counts)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Погибли (0)", "Выжили (1)"],
        survived_counts.values,
        color=["indianred", "seagreen"],
        edgecolor="k"
    )
    plt.bar_label(bars)
    plt.title("Количество выживших и погибших пассажиров")
    plt.ylabel("Количество пассажиров")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

    # 3. Заполнение пропусков в Age медианой
    print(f"\nПропуски в Age до обработки: {df['Age'].isnull().sum()}")
    median_age = df["Age"].median()
    df["Age"] = df["Age"].fillna(median_age)
    print(f"Медиана Age: {median_age}")
    print(f"Пропуски в Age после обработки: {df['Age'].isnull().sum()}")

    # 4. One-Hot Encoding для Sex и Embarked
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)
    print("\nСтолбцы после One-Hot Encoding:")
    print(df.columns.tolist())

    # 5. Гистограмма распределения возрастов
    plt.figure(figsize=(9, 5))
    plt.hist(df["Age"], bins=30, color="royalblue", edgecolor="k", alpha=0.8)
    plt.title("Распределение возрастов пассажиров")
    plt.xlabel("Возраст, лет")
    plt.ylabel("Количество пассажиров")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

    # 6. Новый признак FamilySize
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    print("\nНовый признак FamilySize:")
    print(df[["SibSp", "Parch", "FamilySize"]].head())

    print("\nИтоговый DataFrame:")
    print(df.head())


if __name__ == "__main__":
    main()
