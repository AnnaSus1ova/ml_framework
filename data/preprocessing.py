import numpy as np
from typing import Callable, Any


# Поиск индексов элементов, выходящих за границы (выбросы)
def get_boxplot_outliers(
        data: np.ndarray,
        key: Callable[[Any], Any] = None  # функция для преобразования данных
        ) -> np.ndarray:

    # Сортировка по вертикали
    data_sorted = np.sort(data, axis=0)

    n = len(data_sorted)

    q1 = data_sorted[int(n * 0.25)]

    q3 = data_sorted[int(n * 0.75)]

    # Расчет размаха с коэффициентом 1.5 для определения границ выбросов
    epsilon = (q3 - q1) * 1.5

    lower_bound = q1 - epsilon

    upper_bound = q3 + epsilon

    outliers = np.where((data < lower_bound) | (data > upper_bound))[0]

    return outliers


def train_test_split(
        features: np.ndarray,  # Признаки
        targets: np.ndarray,  # Метки классов
        train_ratio: float = 0.8,  # доля обучающей выборки
        shuffle: bool = True  # флаг перемешивания данных
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if features.shape[0] != targets.shape[0]:
        raise ValueError("Количество объектов в features и "
                         "targets должно совпадать")

    if not (0 <= train_ratio <= 1):
        raise ValueError("train_ratio должен быть в диапазоне (0, 1)")

    train_features, test_features = [], []
    train_targets, test_targets = [], []

    for class_label in np.unique(targets):
        class_mask = (targets == class_label)
        class_features = features[class_mask]
        class_targets = targets[class_mask]

        n_class_samples = len(class_targets)
        n_train = int(round(n_class_samples * train_ratio))

        # Добавление в обучающую выборку
        if n_train > 0:
            train_features.append(class_features[:n_train])
            train_targets.append(class_targets[:n_train])
        # Добавление в тестовую выборку
        if n_class_samples - n_train > 0:
            test_features.append(class_features[n_train:])
            test_targets.append(class_targets[n_train:])

    # Объединение данных всех классов в numpy массивы
    train_features = np.concatenate(train_features) if train_features \
        else np.empty((0, *features.shape[1:]))
    train_targets = np.concatenate(train_targets) if train_targets \
        else np.empty(0, dtype=targets.dtype)
    test_features = np.concatenate(test_features) if test_features \
        else np.empty((0, *features.shape[1:]))
    test_targets = np.concatenate(test_targets) if test_targets \
        else np.empty(0, dtype=targets.dtype)

    if shuffle:
        # Перемешивание обучающей выборки
        train_idx = np.random.permutation(len(train_targets))
        train_features, train_targets = train_features[train_idx], \
            train_targets[train_idx]

        # Перемешивание тестовой выборки
        test_idx = np.random.permutation(len(test_targets))
        test_features, test_targets = test_features[test_idx], \
            test_targets[test_idx]

    return train_features, train_targets, test_features, test_targets
