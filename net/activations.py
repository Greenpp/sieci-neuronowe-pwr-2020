from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass


class Linear(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1


class Sigmoid(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig_x = self(x)

        return sig_x * (1 - sig_x)


class Unipolar(Activation):
    """
    Unipolar activation
    """

    def __init__(self, theta: float = 0):
        self.theta = theta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        activated = np.zeros_like(x)
        activated[x > self.theta] = 1

        return activated

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1


class Bipolar(Activation):
    """
    Bipolar activation
    """

    def __init__(self, theta: float = 0):
        self.theta = theta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        activated = np.full_like(x, -1)
        activated[x > self.theta] = 1

        return activated

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1


ACTIVATIONS = {
    'linear': Linear,
    'unipolar': Unipolar,
    'bipolar': Bipolar,
    'sigmoid': Sigmoid,
}


def get_activation_by_name(name: str) -> type:
    return ACTIVATIONS[name]


if __name__ == '__main__':
    pass
