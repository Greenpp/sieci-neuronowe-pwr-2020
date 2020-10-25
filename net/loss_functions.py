from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, output: np.ndarray, label: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass


class MSE(LossFunction):
    """
    Mean square error loss function
    """

    def __call__(self, output: np.ndarray, label: np.ndarray) -> np.ndarray:
        self.y = output
        self.y_hat = label

        error = ((output - label) ** 2) / 2
        # mean error for batch input
        m_error = error.mean(axis=0)

        return m_error.mean()

    def backward(self) -> np.ndarray:
        delta = self.y - self.y_hat

        return delta


class CrossEntropy(LossFunction):
    def __call__(self, output: np.ndarray, label: np.ndarray) -> np.ndarray:
        EPSILON = 1e-12
        stable_output = np.clip(output, EPSILON, None)

        self.y = stable_output
        self.y_hat = label

        # For labels 0, multiplication with 0 -> skipping
        cross_e = (np.where(label == 1, -np.log(stable_output), 0)).sum(axis=1)

        return cross_e.mean()

    def backward(self) -> np.ndarray:
        # For labels 0, multiplication with 0 -> skipping
        # mean or sum ? facebook paper batch_size * m -> alpha * m
        # return np.where(self.y_hat == 1, -1 / self.y, 0).mean(axis=0)[None, :]

        # Gradient only for combination with softmax
        return self.y - self.y_hat


LOSSES = {
    'mse': MSE,
    'cross-entropy': CrossEntropy,
}


def get_loss_by_name(name: str):
    return LOSSES[name]


if __name__ == "__main__":
    pass
