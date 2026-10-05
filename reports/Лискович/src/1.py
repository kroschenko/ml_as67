import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("auto-mpg.csv", sep=",", na_values='?')
    
    print(df.head(10))
    print(df.dtypes)
    print(df.isnull().sum())
    print(df.describe().T[['mean', '50%', 'std']])
    
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    mean_horsepower = df['horsepower'].mean()
    df['horsepower'] = df['horsepower'].fillna(mean_horsepower)
    
    print(f"Пропуски в horsepower: {df['horsepower'].isnull().sum()}")
        
    df['age'] = 83 - df['model year']
    
    df = pd.get_dummies(df, columns=['origin'], dtype=int)
    
    plt.figure(figsize=(10,6))
    plt.scatter(df['weight'], df['mpg'], alpha=0.7, color='royalblue', edgecolors='k')
    plt.title('Зависимость расхода топлива (mpg) от веса автомобиля (weight)', fontsize=14)
    plt.xlabel('Вес автомобиля (weight)', fontsize=12)
    plt.ylabel('Расход топлива (mpg)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
    plt.figure(figsize=(8, 5))
    cylinder_counts = df["cylinders"].value_counts().sort_index()
    plt.bar(
        cylinder_counts.index,
        cylinder_counts.values,
        edgecolor="k",
        color="salmon"
    )
    plt.title("Распределение количества цилиндров")
    plt.xlabel("Количество цилиндров")
    plt.ylabel("Количество автомобилей")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()
    
    print(f"Осталось пропусков: {df.isnull().sum().sum()}")
    print(df.head(10))

if __name__ == "__main__":
    main()