import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def find_max_fliers_column(df : pd.DataFrame) -> tuple[int, int]:
    fig, ax = plt.subplots()
    box_plot = ax.boxplot(
        df, 
        tick_labels=df.columns.to_list()
    )
    outlier_counts : list[int] = [len(fliers.get_ydata()) for fliers in box_plot['fliers']]
    fliers_max_column : int = int(np.argmax(outlier_counts))
    plt.close()
    return fliers_max_column, outlier_counts[fliers_max_column]

def main() -> None:
    try:
        # Таблица качества проб вина
        wine_quality : pd.DataFrame = pd.read_csv('winequality-white.csv', sep=';')
        print(f"Типы данных столбов:\n{wine_quality.dtypes}")

        # Промежутки, по которым распределяются данные
        qual_bins : list[int | float] = [-float('inf'), 4, 6, float('inf')]
        # Именования промежутков
        qual_labels : list[str] = ['плохое', 'среднее', 'хорошее']

        # Категоризация значений качества вина
        wine_quality["quality"] = pd.cut(
            wine_quality["quality"],
            bins=qual_bins,
            labels=qual_labels,
            right=True,
            include_lowest=True
        ).astype('category')

        # Вычисление коэффициента корреляции между 'fixed acidity' и 'pH' (коэфф. Пирсона)
        corr_value, p_value = stats.pearsonr(wine_quality['fixed acidity'], wine_quality['pH'])
        # Подсчет количества проб каждого типа качества
        quantity_of_qualities : pd.Series = (wine_quality.groupby("quality")
                                 ["quality"].count())
        """Построение столбчатой диаграммы количества вин каждой категории качества"""
        np.random.seed(34)
        colors : np.ndarray = np.random.uniform(15, 80, len(wine_quality['fixed acidity']))
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        fig.canvas.manager.set_window_title("Визуализация характеристик проб вин") # type: ignore
        ax1.bar(qual_labels, 
                quantity_of_qualities.values, 
                width=1, 
                edgecolor="white", 
                linewidth=0.7,
                )
        ax1.set_title("Категории качества")
        
        """"Построение графика корреляции fixed acidity и pH"""
        ax2.scatter(x=wine_quality['fixed acidity'].values, y=wine_quality['pH'].values, c=colors, s=50, vmin=1, vmax=75, alpha=0.5)
        ax2.set(
            xlabel='Fixed acidity',
            ylabel='pH',
            title=f"Корреляция между 'fixed acidity' и 'pH'."
        )
        # Вычисление линии тренда
        z : np.ndarray = np.polyfit(
            wine_quality['fixed acidity'], 
            wine_quality['pH'], 
            1
            )
        p = np.poly1d(z)
        fixed_acidity_interpolation : pd.Series = (wine_quality.loc[
                wine_quality['fixed acidity'].between(
                wine_quality['fixed acidity'].quantile(.0025), 
                wine_quality['fixed acidity'].quantile(.990)),
                'fixed acidity']
                )
        # График линии тренда 
        ax2.plot(fixed_acidity_interpolation.values, 
                 p(fixed_acidity_interpolation), 
                 c='red', 
                 linewidth=2, 
                 label=f"Тренд (k={corr_value:.2f})")
        ax2.legend()
        numeric_col : pd.DataFrame = (wine_quality.
                                        select_dtypes(include='number')
        )
        # Нахождение колонки с максимальным количеством выбросов
        max_fliers_column, count = find_max_fliers_column(numeric_col)
        print(f"Величина '{wine_quality.columns[max_fliers_column]}' имеет наибольшее количество выбросов: {count}")

        plt.show()
        # Стандартизация числовых значений
        for col_name in numeric_col.columns.to_list():
            wine_quality[col_name] = (wine_quality[col_name] - wine_quality[col_name].mean()) / wine_quality[col_name].std()
        print(wine_quality)
    except Exception as a:
        print(a)





if __name__ == "__main__":
    main()