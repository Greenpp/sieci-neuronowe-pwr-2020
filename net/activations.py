from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, grad: np.ndarray) -> np.ndarray:
        pass


class Linear(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return grad


class Sigmoid(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.signal = x
        return 1 / (1 + np.exp(-x))

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        sig_signal = self(self.signal)
        d_sig = sig_signal * (1 - sig_signal)

        return d_sig * grad


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

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return grad


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

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return grad


class Softmax(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Axis 1 for batch input
        stable_x = x - x.max(axis=1)[:, None]
        exp_x = np.exp(stable_x)
        # [:, None] to divide rows not columns
        d_soft = exp_x / exp_x.sum(axis=1)[:, None]

        self.signal = d_soft

        return d_soft

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        return self.signal * (grad - (grad * self.signal).sum(axis=1)[:, None])


class ReLU(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # TODO check if copy is needed
        self.signal = x
        activated = np.clip(x, 0, None)

        return activated

    def derivative(self, grad: np.ndarray) -> np.ndarray:
        # NOTE for optimization grad can replace 1
        d_rel = np.where(self.signal > 0, 1, 0)

        return d_rel * grad


ACTIVATIONS = {
    'linear': Linear,
    'unipolar': Unipolar,
    'bipolar': Bipolar,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'relu': ReLU,
}


def get_activation_by_name(name: str) -> type:
    return ACTIVATIONS[name]


if __name__ == '__main__':
    pass
