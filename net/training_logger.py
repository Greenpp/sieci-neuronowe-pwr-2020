from typing import List

import numpy as np


class TrainingLogger:
    def __init__(self) -> None:
        self.val_errors = []
        self.test_errors = []
        self.train_errors = []
        self.failed = False
        self.weights = []
        self.biases = []
        self.accuracies = []

    def log_val_error(self, error: np.ndarray) -> None:
        self.val_errors.append(error)

    def log_test_error(self, error: np.ndarray) -> None:
        self.test_errors.append(error)

    def log_train_error(self, error: np.ndarray) -> None:
        self.train_errors.append(error)

    def log_accuracy(self, acc: float) -> None:
        self.accuracies.append(acc)

    def log_fail(self) -> None:
        self.failed = True

    def log_weights_and_biases(
        self, weights: List[np.ndarray], biases: List[np.ndarray]
    ) -> None:
        self.weights.append(weights)
        self.biases.append(biases)

    def get_logs(self) -> dict:
        return {
            'val_errors': self.val_errors,
            'test_errors': self.test_errors,
            'train_errors': self.train_errors,
            'epochs': len(self.accuracies)
            - 1,  # first acc log is before training loop
            'failed': self.failed,
            'weights': self.weights,
            'biases': self.biases,
            'accuracies': self.accuracies,
        }
