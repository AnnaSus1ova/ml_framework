import numpy as np
import matplotlib.pyplot as plt
from enum import StrEnum
from typing import Union


class AxisNames(StrEnum):
    X = "x"
    Y = "y"


class DiagramTypes(StrEnum):
    Violin = "violin"
    Hist = "hist"
    Boxplot = "boxplot"


def visualize_distribution(
    points: np.ndarray,
    diagram_type: Union[DiagramTypes, list[DiagramTypes]],
    diagram_axis: Union[AxisNames, list[AxisNames]],
    path_to_save: str = "",
) -> None:

    if not isinstance(diagram_type, list):
        diagram_type = [diagram_type]
    if not isinstance(diagram_axis, list):
        diagram_axis = [diagram_axis]

    plt.style.use('seaborn-v0_8-pastel')
    fig, axes = plt.subplots(
        len(diagram_type),
        len(diagram_axis),
        figsize=(10, 7),
        squeeze=False,
        facecolor='#f5f5f5'
    )

    # Цветовая палитра
    violin_color = '#6a8fd8'
    hist_color = '#88c999'
    box_color = '#f7a072'
    median_color = '#e63946'

    for i, d_type in enumerate(diagram_type):
        for j, axis in enumerate(diagram_axis):
            ax = axes[i, j]
            data = points[:, 0] if axis == AxisNames.X else points[:, 1]

            # Стиль осей
            ax.set_facecolor('#fafafa')
            ax.grid(color='white', linestyle='--', linewidth=0.7)

            # Удаление рамки
            for spine in ax.spines.values():
                spine.set_visible(False)

            if d_type == DiagramTypes.Violin:
                parts = ax.violinplot(
                    data,
                    showmeans=True,
                    showmedians=True,
                    vert=False,
                    widths=0.8
                )

                for pc in parts['bodies']:
                    pc.set_facecolor(violin_color)
                    pc.set_edgecolor('#3a5a9d')
                    pc.set_alpha(0.8)

                parts['cbars'].set_color('#3a5a9d')
                parts['cmins'].set_color('#3a5a9d')
                parts['cmaxes'].set_color('#3a5a9d')
                parts['cmedians'].set_color(median_color)
                parts['cmeans'].set_color('#2a9d8f')

            elif d_type == DiagramTypes.Hist:
                # Стилизация гистограммы
                ax.hist(
                    data,
                    bins=20,
                    color=hist_color,
                    edgecolor='#4a8c5d',
                    alpha=0.85,
                    density=True
                )

                # Добавляем KDE поверх гистограммы
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x = np.linspace(min(data), max(data), 100)
                ax.plot(x, kde(x), color='#2a9d8f', linewidth=2)

            elif d_type == DiagramTypes.Boxplot:
                ax.boxplot(
                    data,
                    vert=False,
                    patch_artist=True,
                    widths=0.5,
                    boxprops=dict(
                        facecolor=box_color,
                        edgecolor='#d46a43',
                        linewidth=1.5
                    ),
                    medianprops=dict(
                        color=median_color,
                        linewidth=2
                    ),
                    whiskerprops=dict(
                        color='#d46a43',
                        linewidth=1.5
                    ),
                    capprops=dict(
                        color='#d46a43',
                        linewidth=1.5
                    ),
                    flierprops=dict(
                        marker='o',
                        markersize=6,
                        markerfacecolor='#e63946',
                        markeredgecolor='none',
                        alpha=0.5
                    )
                )

            ax.set_title(
                f"{d_type.value.upper()} PLOT: {axis.value}-axis",
                fontsize=10,
                pad=12,
                fontweight='bold',
                color='#333333'
            )

            ax.set_xlabel(
                'Values',
                fontsize=9,
                color='#555555'
            )

            ax.tick_params(
                colors='#666666',
                labelsize=8
            )

    fig.suptitle(
        'Data Distribution Analysis',
        y=1.02,
        fontsize=12,
        fontweight='bold',
        color='#333333'
    )

    plt.tight_layout()

    if path_to_save:
        plt.savefig(
            path_to_save,
            dpi=300,
            bbox_inches='tight',
            facecolor=fig.get_facecolor()
        )
    plt.show()
