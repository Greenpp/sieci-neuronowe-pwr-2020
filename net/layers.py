from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from net.activations import Activation


class Layer(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass


class FCLayer(Layer):
    """
    Fully connected layer
    """

    def __init__(
        self, in_: int, out: int, activation: Activation, bias: bool = True
    ) -> None:
        self.weights = np.random.rand(in_, out) - 0.5  #  TODO weights range
        self.input_signal = None
        self.pre_activation_signal = None
        self.activation = activation

        self.bias = bias
        if bias:
            self.b_weights = np.random.rand(1, out) - 0.5  #  TODO bias weights range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute forward pass
        """
        self.input_signal = x

        f_x = x @ self.weights
        if self.bias:
            f_x = f_x + self.b_weights
        self.pre_activation_signal = f_x
        f_x = self.activation(f_x)

        return f_x

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        d_activation = self.activation.derivative(self.pre_activation_signal)
        # Bias delta equal to incoing gradient * activation derivative
        d_b = d_activation * grad
        # Weights delta equal to incoming gradint * activaton derivative * previous layer
        d_w = self.input_signal.T @ d_b

        # Gradient for next layer scaled with weights
        new_grad = d_b @ self.weights.T

        return d_b, d_w, new_grad


if __name__ == '__main__':
    pass
