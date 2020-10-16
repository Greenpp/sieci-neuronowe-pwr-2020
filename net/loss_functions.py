from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, output: np.ndarray, label: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output: np.ndarray, label: np.ndarray) -> np.ndarray:
        pass


class MSE(LossFunction):
    """
    Mean square error loss function
    """

    def __call__(self, output: np.ndarray, label: np.ndarray) -> np.ndarray:
        error = ((output - label) ** 2) / 2
        # mean error for batch input
        m_error = error.mean(axis=0)

        return m_error

    def backward(self, output: np.ndarray, label: np.ndarray) -> np.ndarray:
        delta = output - label

        return delta


class CrossEntropy(LossFunction):
    pass


LOSSES = {
    'mse': MSE,
    'cross-entropy': CrossEntropy,
}


def get_loss_by_name(name: str):
    return LOSSES[name]
