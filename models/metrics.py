import numpy as np


# Оценка качества
def accuracy(true_targets: np.ndarray, prediction: np.ndarray) -> float:
    return np.mean(true_targets == prediction)
