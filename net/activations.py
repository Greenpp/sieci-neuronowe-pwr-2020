from abc import abstractmethod
from typing import Tuple, Type

import numpy as np

from net.model import Layer


class Activation(Layer):
    @abstractmethod
    def derivative(self, grad: np.ndarray) -> np.ndarray:
        pass

    def backward(self, grad: np.ndarray) -> Tuple[None, None, np.ndarray]:
        return None, None, self.derivative(grad)


UNIPOLAR = 'unipolar'
BIPOLAR = 'bipolar'
SIGMOID = 'sigmoid'
SOFTMAX = 'softmax'
SOFTMAX_CE = 'softmax_ce'
RELU = 'relu'
TANH = 'tanh'


class Unipolar(Activation):
    """
    Unipolar activation
    """

    def __init__(self, theta: float = 0):
        self.theta = theta

    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
        activated = np.where(x > self.theta, 1, 0)

        return activated

    def __str__(self) -> str:
        return UNIPOLAR

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return grad


class Bipolar(Activation):
    """
    Bipolar activation
    """

    def __init__(self, theta: float = 0):
        self.theta = theta

    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
        activated = np.where(x > self.theta, 1, -1)

        return activated

    def __str__(self) -> str:
        return BIPOLAR

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return grad


class Sigmoid(Activation):
    """
    Sigmoid activation
    """

    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
        sig = 1 / (1 + np.exp(-x))
        self.cache = sig

        return sig

    def __str__(self) -> str:
        return SIGMOID

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        d_sig = self.cache * (1 - self.cache)

        return d_sig * grad


class Softmax(Activation):
    """
    Softmax activation
    """

    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
        # Axis 1 for batch input
        stable_x = x - x.max(axis=1)[:, None]
        exp_x = np.exp(stable_x)
        # [:, None] to divide rows not columns
        soft = exp_x / exp_x.sum(axis=1)[:, None]

        self.cache = soft

        return soft

    def __str__(self) -> str:
        return SOFTMAX

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        # diagonal = self.cache * np.identity(self.cache.size)
        # d_softmax = diagonal - (self.cache.T @ self.cache)
        # grad @ d_softmax
        # for every batch, optimized

        return self.cache * (grad - (grad * self.cache).sum(axis=1)[:, None])


class SoftmaxCE(Softmax):
    """
    Softmax activation for last layer before cross-entropy loss function
    """

    def __str__(self) -> str:
        return SOFTMAX_CE

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return grad


class ReLU(Activation):
    """
    ReLU activation
    """

    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
        self.cache = x
        activated = np.clip(x, 0, None)

        return activated

    def __str__(self) -> str:
        return RELU

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        d_rel = np.where(self.cache > 0, grad, 0)

        return d_rel


class TanH(Activation):
    """
    TanH activation
    """

    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
        sig = (2 / (np.exp(-2 * x) + 1)) - 1
        self.cache = sig

        return sig

    def __str__(self) -> str:
        return TANH

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        d_tan = 1 - (self.cache ** 2)

        return d_tan * grad


ACTIVATIONS = {
    UNIPOLAR: Unipolar,
    BIPOLAR: Bipolar,
    SIGMOID: Sigmoid,
    SOFTMAX: Softmax,
    SOFTMAX_CE: SoftmaxCE,
    RELU: ReLU,
    TANH: TanH,
}


def get_activation_by_name(name: str) -> Type[Activation]:
    return ACTIVATIONS[name]


if __name__ == '__main__':
    pass
