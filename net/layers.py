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
    def __str__(self) -> str:
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        pass


FC = 'fc'


class FCLayer(Layer):
    """
    Fully connected layer
    """

    def __init__(
        self,
        in_: int = 1,
        out: int = 1,
        activation: Activation = None,
        bias: bool = True,
        weight_range: Tuple[float, float] = (-0.5, 0.5),
    ) -> None:
        self.weights = self._init_weights((in_, out), weight_range)
        self.input_signal = None
        self.activation = activation

        self.bias = bias
        if bias:
            self.b_weights = self._init_weights((1, out), weight_range)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute forward pass
        """
        self.input_signal = x

        f_x = x @ self.weights
        if self.bias:
            f_x = f_x + self.b_weights
        f_x = self.activation(f_x)

        return f_x

    def __str__(self) -> str:
        return FC

    def _init_weights(
        self, shape: Tuple[int, int], range_: Tuple[float, float]
    ) -> np.ndarray:
        # Random 0 - 1
        weights = np.random.rand(*shape)
        # Shift to range
        min_, max_ = range_
        size = max_ - min_

        return weights * size + min_

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # d_b = activation delta * grad
        # TODO rename to backward
        d_b = self.activation.derivative(grad)

        # Weights delta equal to incoming gradint * activaton derivative * previous layer
        d_w = self.input_signal.T @ d_b

        # Gradient for next layer scaled with weights
        new_grad = d_b @ self.weights.T

        # Accumulate bias delta for batch input
        acc_d_b = d_b.sum(axis=0)

        return acc_d_b, d_w, new_grad


LAYERS = {FC: FCLayer}


def get_layer_by_name(name: str) -> type:
    return LAYERS[name]


if __name__ == '__main__':
    pass
