import numpy as np
from typing import Callable


# Функция вычисления евклидова расстояния между векторами
def euclidean_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Вычисляем сумму квадратов разностей и извлекаем корень (по каждой строке)
    return np.sqrt(np.sum((a - b)**2, axis=1))


class KNearestNeighbors:
    def __init__(self, n_neighbors: int = 5, calc_distances:
                 Callable = euclidean_dist):
        # Количество соседей для классификации
        self.n_neighbors = n_neighbors
        # Функция для вычисления расстояний
        self.calc_distances = calc_distances

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Сохраняем обучающие данные
        self.X_train = X_train
        # Сохраняем метки классов
        self.y_train = y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # Вычисляем матрицу расстояний между тестовыми и обучающими примерами
        # Используем broadcasting для попарных вычислений
        distances = np.sqrt(((X_test[:, np.newaxis] -
                              self.X_train) ** 2).sum(axis=2))

        # Определяем фактическое количество соседей (не больше чем доступно)
        actual_n_neighbors = min(self.n_neighbors, len(self.X_train))

        # Получаем индексы ближайших соседей
        if actual_n_neighbors == len(self.X_train):
            # Если нужно использовать все точки - создаем индексы через tile
            nearest_indices = np.tile(np.arange(len(self.X_train)),
                                      (len(X_test), 1))
        else:
            # Иначе используем argpartition для эффективного поиска k ближайших
            nearest_indices = np.argpartition(distances, actual_n_neighbors-1,
                                              axis=1)[:, :actual_n_neighbors]

        # Получаем метки ближайших соседей
        nearest_labels = self.y_train[nearest_indices]

        # Находим уникальные классы в обучающей выборке
        unique_classes = np.unique(self.y_train)

        # Создаем one-hot encoded матрицу голосов
        # Сравниваем метки с каждым классом через broadcasting
        votes = (nearest_labels[:, :, np.newaxis] ==
                 unique_classes[np.newaxis, np.newaxis, :])

        # Суммируем голоса для каждого класса
        vote_counts = votes.sum(axis=1)

        # Выбираем класс с максимальным количеством голосов
        predictions = unique_classes[np.argmax(vote_counts, axis=1)]

        return predictions


class WeightedKNearestNeighbors:
    def __init__(self, n_neighbors: int = 5, calc_distances:
                 Callable = euclidean_dist):
        # Количество соседей для классификации
        self.n_neighbors = n_neighbors
        # Функция для вычисления расстояний
        self.calc_distances = calc_distances

    # Функция ядра Епанечникова для взвешивания
    def _epanechnikov_kernel(self, x):
        # Возвращает вес по правилу: 0.75*(1-x^2) если |x| <=1, иначе 0
        return np.where(np.abs(x) <= 1, 0.75 * (1 - x**2), 0)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Сохраняем обучающие данные
        self.X_train = X_train
        # Сохраняем метки классов
        self.y_train = y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # 1. Вычисляем матрицу расстояний от всех тестовых
        # точек до всех обучающих
        distances = np.array([self.calc_distances(self.X_train, x)
                              for x in X_test])

        # 2. Находим индексы k ближайших соседей для каждой точки
        nearest_indices = np.argpartition(distances, self.n_neighbors-1,
                                          axis=1)[:, :self.n_neighbors]

        # 3. Находим расстояние до самого дальнего соседа (h) для каждой точки
        h = np.take_along_axis(distances, nearest_indices, axis=1)[:, -1:]

        # 4. Вычисляем веса через ядро Епанечникова (нормируем расстояния)
        scaled_dists = np.take_along_axis(distances, nearest_indices,
                                          axis=1) / h
        weights = self._epanechnikov_kernel(scaled_dists)

        # 5. Получаем метки ближайших соседей
        nearest_labels = self.y_train[nearest_indices]

        # 6. Создаем маску для каждого класса
        unique_classes = np.unique(self.y_train)
        class_mask = (nearest_labels[:, :, np.newaxis] ==
                      unique_classes[np.newaxis, np.newaxis, :])

        # 7. Взвешенное голосование - умножаем веса на маску и суммируем
        weighted_votes = np.sum(weights[:, :, np.newaxis] * class_mask, axis=1)

        # 8. Выбираем класс с максимальным весом
        predictions = unique_classes[np.argmax(weighted_votes, axis=1)]

        return predictions
