from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, output: np.ndarray, label: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass


class MSE(LossFunction):
    """
    Mean square error loss function
    """

    def __call__(self, output: np.ndarray, label: np.ndarray) -> float:
        self.y = label
        self.y_hat = output

        error = ((output - label) ** 2) / 2
        mean_err = error.mean()

        return mean_err

    def backward(self) -> np.ndarray:
        delta = self.y_hat - self.y

        return delta


class CrossEntropy(LossFunction):
    """
    Cross-Entropy loss function
    """

    def __call__(self, output: np.ndarray, label: np.ndarray) -> float:
        # Prevent log of 0
        EPSILON = 1e-12
        stable_output = np.clip(output, EPSILON, None)

        self.y = label
        self.y_hat = stable_output

        # For labels 0, multiplication with 0 -> skipping
        cross_e = (np.where(label == 1, -np.log(stable_output), 0)).sum(axis=1)

        return cross_e.mean()

    def backward(self) -> np.ndarray:
        # For labels 0, multiplication with 0 -> skipping
        # mean or sum ? facebook paper batch_size * m -> alpha * m
        # return np.where(self.y_hat == 1, -1 / self.y, 0).mean(axis=0)[None, :]

        # Gradient only for combination with softmax
        delta = self.y_hat - self.y
        return delta


LOSSES = {
    'mse': MSE,
    'cross-entropy': CrossEntropy,
}


def get_loss_by_name(name: str):
    return LOSSES[name]


if __name__ == "__main__":
    pass
