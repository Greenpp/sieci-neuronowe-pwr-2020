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

    # TODO custom theta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[x > 0] = 1
        x[x < 0] = 0

        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1


class Bipolar(Activation):
    """
    Bipolar activation
    """

    # TODO custom theta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[x > 0] = 1
        x[x < 0] = -1

        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1


if __name__ == "__main__":
    pass
