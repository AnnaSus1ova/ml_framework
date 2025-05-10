import sklearn.datasets as skd
import sys
from pathlib import Path
from data.visualization import visualize_distribution, AxisNames, DiagramTypes
from data.preprocessing import get_boxplot_outliers, train_test_split
from models.knn import KNearestNeighbors, WeightedKNearestNeighbors
from models.metrics import accuracy
from tools.animations import AnimationKNN

# Добавляем корень проекта в sys.path
project_root = Path(__file__).parent.parent  # Поднимаемся на 2 уровня вверх
sys.path.append(str(project_root))

# Генерация данных
points, labels = skd.make_moons(n_samples=400, noise=0.3)

# Визуализация распределений
visualize_distribution(
    points,
    diagram_type=[DiagramTypes.Violin, DiagramTypes.Hist,
                  DiagramTypes.Boxplot],
    diagram_axis=[AxisNames.X, AxisNames.Y],
    path_to_save="images/distribution.png"
)

# Поиск выбросов
outliers = get_boxplot_outliers(points)
print(f"Found {len(outliers)} outliers in points")

# Разделение на train/test
X_train, y_train, X_test, y_test = train_test_split(points, labels)

# Обучение моделей
knn = KNearestNeighbors(n_neighbors=5)
knn.fit(X_train, y_train)

wknn = WeightedKNearestNeighbors(n_neighbors=5)
wknn.fit(X_train, y_train)

# Предсказания и оценка качества
knn_pred = knn.predict(X_test)
wknn_pred = wknn.predict(X_test)

print(f"KNN Accuracy: {accuracy(y_test, knn_pred):.2f}")
print(f"Weighted KNN Accuracy: {accuracy(y_test, wknn_pred):.2f}")

# Создание анимации
animator = AnimationKNN()
animation = animator.create_animation(
    knn, X_test[:20], y_test[:20],
    path_to_save="images/knn_animation.gif"
)

animator = AnimationKNN()
animation = animator.create_animation(
    wknn, X_test[:20], y_test[:20],
    path_to_save="images/wknn_animation.gif"
)
