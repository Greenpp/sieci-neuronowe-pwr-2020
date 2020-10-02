from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_weight(self, error: np.ndarray, alpha: float) -> None:
        pass


class FCLayer(Layer):
    """
    Fully connected layer
    """

    def __init__(self, in_: int, out: int, activation: Callable, bias: bool = True) -> None:
        self.weights = np.random.rand(in_, out)
        self.activation = activation
        self.signal = None

        self.bias = bias
        if bias:
            self.b_weights = np.random.rand(1, out)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute forward pass
        """
        self.signal = x

        f_x = x @ self.weights
        if self.bias:
            f_x = f_x + self.b_weights

        f_x = self.activation(f_x)

        return f_x

    def update_weight(self, error: np.ndarray, alpha: float) -> None:
        """
        Update weights
        """
        delta_w = self.signal.T @ error
        self.weights = self.weights - alpha * delta_w

        if self.bias:
            self.b_weights = self.b_weights - alpha * error


if __name__ == "__main__":
    pass
