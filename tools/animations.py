import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from models.metrics import accuracy


class AnimationKNN:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

    def create_animation(
        self,
        knn,
        X_test: np.ndarray,
        true_targets: np.ndarray,
        path_to_save: str = "",
    ) -> FuncAnimation:
        predictions = knn.predict(X_test)

        def update(frame):
            self.ax.clear()
            x, y = X_test[frame]
            pred = predictions[frame]
            true = true_targets[frame]

            # Отображаем обучающие данные
            self.ax.scatter(
                knn.X_train[:, 0], knn.X_train[:, 1],
                c=knn.y_train, alpha=0.3
            )

            # Отображаем тестовую точку
            color = 'green' if pred == true else 'red'
            self.ax.scatter([x], [y], c=color, s=100)

            # Отображаем ближайших соседей
            distances = knn.calc_distances(knn.X_train, X_test[frame])
            nearest_indices = np.argsort(distances)[:knn.n_neighbors]
            self.ax.scatter(
                knn.X_train[nearest_indices, 0],
                knn.X_train[nearest_indices, 1],
                edgecolors='black', linewidths=1.5, s=80
            )

            self.ax.set_title(
                f"Sample {frame}: Predicted {pred}, Actual {true}\n"
                f"Accuracy: {accuracy(true_targets[:frame+1],
                                      predictions[:frame+1]):.2f}"
            )

        anim = FuncAnimation(
            self.fig, update, frames=len(X_test),
            interval=500, repeat=False
        )

        if path_to_save:
            anim.save(path_to_save, writer='pillow')

        return anim
