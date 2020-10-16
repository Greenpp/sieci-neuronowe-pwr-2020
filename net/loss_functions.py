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
    def __call__(self, output: np.ndarray, label: np.ndarray) -> np.ndarray:
        EPSILON = 1e-12
        stable_output = np.clip(output, EPSILON, 1.0 - EPSILON)
        N = stable_output.shape[0]
        cross_e = -np.sum(label * np.log(stable_output + 1e-9)) / N

        return cross_e

    def backward(self) -> np.ndarray:
    pass


LOSSES = {
    'mse': MSE,
    'cross-entropy': CrossEntropy,
}


def get_loss_by_name(name: str):
    return LOSSES[name]
