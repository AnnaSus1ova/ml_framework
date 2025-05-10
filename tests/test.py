import numpy as np
import pytest
from data.preprocessing import get_boxplot_outliers, train_test_split
from models.knn import euclidean_dist, KNearestNeighbors, \
    WeightedKNearestNeighbors


"""
Фикстуры для тестовых данных:
- Фикстуры pytest используются для подготовки данных перед тестами
- Позволяют избежать дублирования кода инициализации данных
"""


@pytest.fixture
def outlier_data():
    """Тестовые данные с явными выбросами"""
    return np.array([
        [1, 1.2],
        [2, 1.0],
        [3, 0.8],
        [4, -10.0],  # выброс по Y
        [50, -0.5],  # выброс по X
        [5, 100.0]   # выброс по Y
    ])


@pytest.fixture
def classification_data():
    """Тестовые данные для проверки разделения на train/test"""
    np.random.seed(42)
    features = np.random.rand(100, 2)  # 100 точек с 2 признаками
    targets = np.array([0]*70 + [1]*30)  # 70 нулей и 30 единиц
    return features, targets


"""
Класс тестов для функции обнаружения выбросов get_boxplot_outliers
"""


class TestGetBoxplotOutliers:
    def test_outlier_detection(self, outlier_data):
        """Проверка корректного обнаружения выбросов"""
        outliers = get_boxplot_outliers(outlier_data)
        assert len(outliers) == 3  # Должно найти 3 выброса
        assert set(outliers) == {3, 4, 5}  # Конкретные индексы выбросов

    def test_no_outliers(self):
        """Проверка на данных без выбросов - должен вернуть пустой список"""
        data = np.array([[1, 2], [2, 3], [3, 4]])
        outliers = get_boxplot_outliers(data)
        assert len(outliers) == 0


"""
Класс тестов для функции разделения данных train_test_split
"""


class TestTrainTestSplit:
    def test_split_sizes(self, classification_data):
        """Проверка корректности размеров train и test выборок"""
        features, targets = classification_data
        train_feats, _, test_feats, _ = train_test_split(
            features, targets, train_ratio=0.7
        )
        assert len(train_feats) == 70
        assert len(test_feats) == 30

    def test_class_balance(self, classification_data):
        """Проверка сохранения баланса классов в train и test выборках"""
        features, targets = classification_data
        _, train_lbls, _, test_lbls = train_test_split(
            features, targets, train_ratio=0.7
        )

        # Проверяем пропорции классов в train (70/30)
        assert np.isclose(
            sum(train_lbls == 0) / sum(train_lbls == 1),
            70 / 30,
            rtol=0.1  # Допустимое отклонение 10%
        )

        # Проверяем пропорции классов в test (должны быть такие же)
        assert np.isclose(
            sum(test_lbls == 0) / sum(test_lbls == 1),
            70 / 30,
            rtol=0.1
        )

    def test_random_seed(self, classification_data):
        """Проверка воспроизводимости результатов при одинаковом random seed"""
        features, targets = classification_data

        # Два вызова с одинаковым seed должны дать одинаковые результаты
        np.random.seed(42)
        train1, _, _, _ = train_test_split(features, targets)

        np.random.seed(42)
        train2, _, _, _ = train_test_split(features, targets)

        assert np.array_equal(train1, train2)


"""
Дополнительные тесты для обработки некорректных входных данных
"""


def test_invalid_inputs():
    """Проверка обработки невалидных входных данных"""
    with pytest.raises(ValueError):
        # Разная длина features и targets
        train_test_split(np.array([[1, 2]]), np.array([1, 2]))

    with pytest.raises(ValueError):
        # Некорректный train_ratio (>1)
        train_test_split(np.array([[1, 2]]), np.array([1]), train_ratio=1.5)


"""
Тесты для функции вычисления евклидова расстояния
"""


def test_euclidean_dist():
    # Проверка 2D случая (теорема Пифагора)
    assert np.allclose(
        euclidean_dist(np.array([[0, 0]]), np.array([[3, 4]])),
        [5.0]
    )
    # Проверка 3D случая
    assert np.allclose(
        euclidean_dist(np.array([[1, 1, 1]]), np.array([[4, 5, 6]])),
        [np.sqrt(9+16+25)]
    )


"""
Фикстура для тестов KNN - простой двумерный набор данных с двумя классами
"""


@pytest.fixture
def knn_dataset():
    X_train = np.array([
        [0.0, 0.0],  # Класс 0
        [1.0, 1.0],  # Класс 0
        [2.0, 2.0],   # Класс 0
        [10.0, 10.0],  # Класс 1
        [11.0, 11.0]  # Класс 1
    ])
    y_train = np.array([0, 0, 0, 1, 1])
    X_test = np.array([[1.5, 1.5], [10.5, 10.5]])
    # Тестовые точки между классами
    return X_train, y_train, X_test


"""
Класс тестов для базового алгоритма K ближайших соседей
"""


class TestKNearestNeighbors:
    def test_knn_basic_prediction(self, knn_dataset):
        """Тест базового предсказания KNN"""
        X_train, y_train, X_test = knn_dataset
        knn = KNearestNeighbors(n_neighbors=3)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        assert len(preds) == 2
        assert preds[0] == 0  # Ближе к классу 0
        assert preds[1] == 1  # Ближе к классу 1

    def test_knn_single_point(self):
        """Тест на наборе с одной точкой"""
        knn = KNearestNeighbors(n_neighbors=1)
        knn.fit(np.array([[0, 0]]), np.array([42]))
        pred = knn.predict(np.array([[1, 1]]))
        assert pred[0] == 42  # Должен вернуть метку единственной точки


"""
Класс тестов для взвешенного KNN
"""


class TestWeightedKNearestNeighbors:
    def test_weighted_knn_basic_prediction(self, knn_dataset):
        """Тест базового предсказания взвешенного KNN"""
        X_train, y_train, X_test = knn_dataset
        knnw = WeightedKNearestNeighbors(n_neighbors=3)
        knnw.fit(X_train, y_train)
        preds = knnw.predict(X_test)
        assert len(preds) == 2
        assert preds[0] == 0  # Ближе к классу 0
        assert preds[1] == 1  # Ближе к классу 1

    def test_weighted_knn_equal_distance(self):
        """Тест случая, когда точки на равном расстоянии"""
        X_train = np.array([[0, 0], [2, 0]])  # Точки на одинаковом расстоянии
        y_train = np.array([0, 1])
        X_test = np.array([[1, 0]])
        knnw = WeightedKNearestNeighbors(n_neighbors=2)
        knnw.fit(X_train, y_train)
        pred = knnw.predict(X_test)
        assert pred[0] in (0, 1)  # Может выбрать любой класс

    def test_kernel_output(self):
        """Тест ядра Епанечникова"""
        u = np.array([-1.5, -1, 0, 0.5, 1, 1.5])
        expected = np.array([0, 0, 0.75, 0.5625, 0, 0])

        wknn = WeightedKNearestNeighbors()
        result = np.array([wknn._epanechnikov_kernel(x) for x in u])

        assert np.allclose(result, expected)  # Проверка значений ядра


def test_large_k_value():
    """Тест случая, когда k больше количества точек"""
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])
    knn = KNearestNeighbors(n_neighbors=len(X)+1)
    knn.fit(X, y)
    # Должен использовать все точки и вернуть наиболее частый класс
    assert knn.predict(np.array([[10]]))[0] == 0


def test_knn_with_string_labels():
    """Тест работы с нечисловыми метками классов"""
    X = np.array([[1], [2], [3], [4]])
    y = np.array(["cat", "cat", "dog", "dog"])
    knn = KNearestNeighbors(n_neighbors=2)
    knn.fit(X, y)
    assert knn.predict(np.array([[1.5]]))[0] == "cat"  # Должен вернуть "cat"


def test_extreme_train_ratios():
    """Тест крайних значений train_ratio (0 и 1)"""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1, 2, 3, 4])

    # Все данные в train
    X_train, _, X_test, _ = train_test_split(X, y, train_ratio=1.0)
    assert len(X_train) == 4 and len(X_test) == 0

    # Все данные в test
    X_train, _, X_test, _ = train_test_split(X, y, train_ratio=0.0)
    assert len(X_train) == 0 and len(X_test) == 4


def test_identical_points():
    """Тест обработки идентичных точек с разными метками"""
    X = np.array([[1], [1], [2], [2]])
    y = np.array([0, 1, 0, 1])  # Противоречивые метки
    knn = KNearestNeighbors(n_neighbors=2)
    knn.fit(X, y)
    pred = knn.predict(np.array([[1]]))
    assert pred[0] in (0, 1)  # Случайный выбор при равных расстояниях


def test_model_state_consistency():
    """Тест неизменности модели после предсказания"""
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])
    knn = KNearestNeighbors(n_neighbors=1)
    knn.fit(X, y)

    original_X = knn.X_train.copy()
    original_y = knn.y_train.copy()

    knn.predict(np.array([[1.5]]))  # Делаем предсказание

    # Проверяем, что обучающие данные не изменились
    assert np.array_equal(knn.X_train, original_X)
    assert np.array_equal(knn.y_train, original_y)
