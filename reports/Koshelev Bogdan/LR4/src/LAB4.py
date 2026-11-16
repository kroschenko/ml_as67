import matplotlib.pyplot as plt
import numpy as np

# Данные из вашей таблицы для первых 3 шагов
steps = [
   {"iter": 1, "a": 1.55760, "b": 1.69280, "c": 1.62520, "delta": 1.59140, "drop": "right"},
   {"iter": 2, "a": 1.55760, "b": 1.62520, "c": 1.59140, "delta": 1.60228, "drop": "left"},
   {"iter": 3, "a": 1.59140, "b": 1.62520, "c": 1.60228, "delta": 1.59780, "drop": "right"}
]

fig, axes = plt.subplots(3, 1, figsize=(10, 6))
fig.subplots_adjust(hspace=0.8)

for i, step in enumerate(steps):
    ax = axes[i]
    a, b, c, delta = step["a"], step["b"], step["c"], step["delta"]
    
    # Рисуем линию отрезка
    ax.plot([a, b], [0, 0], color='black', linewidth=3, zorder=1)
    
    # --- ШТРИХОВКА ОТБРОШЕННОЙ ЧАСТИ ---
    if step["drop"] == "right":
        # Отбрасываем правую часть (от c до b)
        ax.axvspan(c, b, color='red', alpha=0.2, hatch='///', zorder=0)
        ax.text((c+b)/2, 0.05, "отброшено", ha='center', color='red', fontsize=10, alpha=0.7)
    else:
        # Отбрасываем левую часть (от a до c)
        ax.axvspan(a, c, color='red', alpha=0.2, hatch='///', zorder=0)
        ax.text((a+c)/2, 0.05, "отброшено", ha='center', color='red', fontsize=10, alpha=0.7)

    # Рисуем точки
    ax.scatter([a, b], [0, 0], color='black', s=80, zorder=2, label='Границы (a, b)')
    ax.scatter([c], [0], color='blue', s=80, zorder=2, label='Середина (c)')
    ax.scatter([delta], [0], color='red', marker='*', s=200, zorder=3, label='Точка параболы (δ)')
    
    # Подписи к точкам
    ax.text(a, -0.15, f"a={a:.4f}", ha='center', fontsize=11, fontweight='bold')
    ax.text(b, -0.15, f"b={b:.4f}", ha='center', fontsize=11, fontweight='bold')
    ax.text(c, 0.1, f"c={c:.4f}", ha='center', color='blue', fontsize=10)
    ax.text(delta, 0.1, f"δ={delta:.4f}", ha='center', color='red', fontsize=10, fontweight='bold')

    # Оформление оси
    ax.set_title(f"Итерация {step['iter']}", loc='left', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.3, 0.2)
    ax.set_xlim(1.54, 1.70) # Чуть расширим границы, чтобы подписи не обрезались
    ax.axis('off') # Убираем лишние рамки и оси Y

# Добавляем легенду только для первого графика
axes[0].legend(loc='upper right', fontsize=9, framealpha=1)

plt.suptitle("Графическая интерпретация метода квадратичной интерполяции", fontsize=14, fontweight='bold')
plt.savefig('graphic_interpretation.png', dpi=300, bbox_inches='tight')
plt.show()