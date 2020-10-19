from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def derivative(self, grad: np.ndarray) -> np.ndarray:
        pass


LINEAR = 'linear'


class Linear(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def __str__(self) -> str:
        return LINEAR

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return grad


SIGMOID = 'sigmoid'


class Sigmoid(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        sig = 1 / (1 + np.exp(-x))
        self.signal = sig

        return sig

    def __str__(self) -> str:
        return SIGMOID

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        d_sig = self.signal * (1 - self.signal)

        return d_sig * grad


UNIPOLAR = 'unipolar'


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

    def __str__(self) -> str:
        return UNIPOLAR

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return grad


BIPOLAR = 'bipolar'


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

    def __str__(self) -> str:
        return BIPOLAR

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return grad


SOFTMAX = 'softmax'


class Softmax(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Axis 1 for batch input
        stable_x = x - x.max(axis=1)[:, None]
        exp_x = np.exp(stable_x)
        # [:, None] to divide rows not columns
        soft = exp_x / exp_x.sum(axis=1)[:, None]

        self.signal = soft

        return soft

    def __str__(self) -> str:
        return SOFTMAX

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        # diagonal = self.signal * np.identity(self.signal.size)
        # d_softmax = diagonal - (self.signal.T @ self.signal)
        # grad @ d_softmax
        # for every batch, optimized

        return self.signal * (grad - (grad * self.signal).sum(axis=1)[:, None])


SOFTMAX_CE = 'softmax_ce'


class SoftmaxCE(Softmax):
    def __str__(self) -> str:
        return SOFTMAX_CE

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return grad


RELU = 'relu'


class ReLU(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.signal = x
        activated = np.clip(x, 0, None)

        return activated

    def __str__(self) -> str:
        return RELU

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        d_rel = np.where(self.signal > 0, grad, 0)

        return d_rel


TANH = 'tanh'


class TanH(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        sig = (2 / (np.exp(-2 * x) + 1)) - 1
        self.signal = sig

        return sig

    def __str__(self) -> str:
        return TANH

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return 1 - (self.signal ** 2)


ACTIVATIONS = {
    LINEAR: Linear,
    UNIPOLAR: Unipolar,
    BIPOLAR: Bipolar,
    SIGMOID: Sigmoid,
    SOFTMAX: Softmax,
    SOFTMAX_CE: SoftmaxCE,
    RELU: ReLU,
    TANH: TanH,
}


def get_activation_by_name(name: str) -> type:
    return ACTIVATIONS[name]


if __name__ == '__main__':
    pass
