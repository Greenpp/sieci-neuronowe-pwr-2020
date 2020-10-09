from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, output: np.ndarray, label: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output: np.ndarray, label: np.ndarray):
        pass


class MSE(LossFunction):
    """
    Mean square error loss function
    """

    def __call__(self, output: np.ndarray, label: np.ndarray) -> np.ndarray:
        # error = ((label - output) ** 2).mean(axis=1) / 2
        # TODO check for more batch
        error = ((output - label) ** 2) / 2

        return error

    def backward(self, output: np.ndarray, label: np.ndarray):
        return output - label
