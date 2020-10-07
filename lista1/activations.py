from abc import ABC, abstractmethod
import numpy as np


def a_unipolar(x: np.ndarray) -> np.ndarray:
    """
    Unipolar activation
    """
    val = x.copy()
    val[val < 0] = 0
    val[val > 0] = 1

    return val


def a_bipolar(x: np.ndarray) -> np.ndarray:
    """
    Bipolar activation
    """
    val = x.copy()
    val[val < 0] = -1
    val[val > 0] = 1

    return val


class Activation(ABC):
    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass


class Unipolar(Activation):
    """
    Unipolar activation
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[x > 0] = 1
        x[x < 0] = 0

        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x


class Bipolar(Activation):
    """
    Bipolar activation
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[x > 0] = 1
        x[x < 0] = -1

        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x


if __name__ == "__main__":
    pass
