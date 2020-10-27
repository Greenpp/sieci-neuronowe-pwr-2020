from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.core.fromnumeric import size
from net.utils import col2im_indices, im2col_indices
from typing import TYPE_CHECKING, Iterable, Tuple

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
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def _init_weights(
        self, shape: Iterable[int], range_: Tuple[float, float]
    ) -> np.ndarray:
        """
        Creates random weights with <shape> in <range_> using normal distribution
        """
        # Random 0 - 1
        weights = np.random.rand(*shape)
        # Shift to range
        min_, max_ = range_
        size = max_ - min_

        return weights * size + min_


FC = 'fc'
CONV = 'conv'
MAXPOLL = 'maxpoll'
FLATTEN = 'flatten'


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
        self.bias = bias
        if bias:
            self.b_weights = self._init_weights((1, out), weight_range)

        self.activation = activation
        self.input_signal = None

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

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # d_b = activation delta * grad
        d_b = self.activation.backward(grad)

        # Weights delta equal to incoming gradint * activaton derivative * previous layer
        d_w = self.input_signal.T @ d_b

        # Gradient for next layer scaled with weights
        new_grad = d_b @ self.weights.T

        # Accumulate bias delta for batch input
        acc_d_b = d_b.sum(axis=0)

        return acc_d_b, d_w, new_grad


class ConvLayer(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Activation = None,
        kernel_size: int = 3,
        bias: bool = True,
        stride: int = 1,
        padding: int = 1,
        weight_range: Tuple[float, float] = (-0.5, 0.5),
    ) -> None:
        kernel_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weights = self._init_weights(kernel_shape, weight_range)
        self.bias = bias
        if bias:
            self.b_weights = self._init_weights((out_channels, 1), weight_range)

        self.activation = activation
        self.stride = stride
        self.padding = padding

        self.filters = out_channels

        self.kernel_size = kernel_size

        self.input_signal = None
        self.input_signal_col = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        batch_size, x_channels, x_height, x_width = x.shape

        # New shape calculated
        x_height_out = (
            x_height - self.kernel_size + 2 * self.padding
        ) / self.stride + 1
        x_width_out = (x_width - self.kernel_size + 2 * self.padding) / self.stride + 1

        if not x_height_out.is_integer() or not x_width_out.is_integer():
            raise Exception('Wrong convolution shape')

        x_height_out = int(x_height_out)
        x_width_out = int(x_width_out)

        # Input reshaped into columns with im2col
        x_col = im2col_indices(
            x, self.kernel_size, self.kernel_size, self.padding, self.stride
        )

        kernel_col = self.weights.reshape(self.filters, -1)

        self.input_signal_col = x_col
        self.input_signal = x

        f_x = kernel_col @ x_col
        if self.bias:
            f_x = f_x + self.bias
        # Reshape to initial form
        f_x = f_x.reshape(self.filters, x_height_out, x_width_out, batch_size)
        f_x = f_x.transpose(3, 0, 1, 2)

        f_x = self.activation(f_x)

        return f_x

    def __str__(self) -> str:
        return CONV

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO what is going on ...
        d_a = self.activation.backward(grad)

        d_b = np.sum(d_a, axis=(0, 2, 3))
        d_b = d_b.reshape(self.filters, -1)

        grad_col = d_a.transpose(1, 2, 3, 0).reshape(self.filters, -1)
        d_w = grad_col @ self.input_signal_col.T
        d_w = d_w.reshape(self.weights.shape)

        w_col = self.weights.reshape(self.filters, -1)
        new_grad_col = w_col.T @ grad_col
        new_grad = col2im_indices(
            new_grad_col,
            self.input_signal.shape,
            self.kernel_size,
            self.kernel_size,
            self.padding,
            self.stride,
        )

        return d_b, d_w, new_grad


class MaxPollLayer(Layer):
    def __init__(self, size: int = 2, stride: int = 2, padding: int = 0) -> None:
        self.size = size
        self.stride = stride
        self.padding = padding

        self.max_idx = None
        self.input_signal = None
        self.input_signal_col = None

    def __call__(self, x: np.ndarray) -> np.ndarray:

        batch_size, channels, x_height, x_width = x.shape
        out_h = (x_height - self.size + 2 * self.padding) / self.stride + 1
        out_w = (x_width - self.size + 2 * self.padding) / self.stride + 1

        if not out_h.is_integer() or not out_w.is_integer():
            raise Exception('Wrong poll shape')

        out_h = int(out_h)
        out_w = int(out_w)

        x_reshaped = x.reshape(batch_size * channels, 1, x_height, x_width)

        x_col = im2col_indices(
            x_reshaped, self.size, self.size, self.padding, self.stride
        )

        max_idx = x_col.argmax(axis=0)

        self.input_signal_col = x_col
        self.input_signal = x
        self.max_idx = max_idx

        out = x_col[max_idx, range(max_idx.size)]
        out = out.reshape(out_h, out_w, batch_size, channels)
        out = out.transpose(2, 3, 0, 1)

        return out

    def __str__(self) -> str:
        return MAXPOLL

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size, channels, x_height, x_width = self.input_signal.shape
        new_grad_col = np.zeros_like(self.input_signal_col)

        flat_grad = grad.transpose(2, 3, 0, 1).ravel()

        new_grad_col[self.max_idx, range(self.max_idx.size)] = flat_grad

        new_grad = col2im_indices(
            new_grad_col,
            (batch_size * channels, 1, x_height, x_width),
            self.size,
            self.size,
            self.padding,
            self.stride,
        )

        new_grad = new_grad.reshape(self.input_signal.shape)

        return None, None, new_grad


class FlattenLayer(Layer):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        batch_size = x.shape[0]

        # TODO check if will work without ravel
        return x.ravel().reshape(batch_size, -1)

    def __str__(self) -> str:
        return FLATTEN

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        new_grad = grad.reshape(self.input_shape)

        return None, None, new_grad


LAYERS = {FC: FCLayer, CONV: ConvLayer, MAXPOLL: MaxPollLayer, FLATTEN: FlattenLayer}


def get_layer_by_name(name: str) -> type:
    return LAYERS[name]


if __name__ == '__main__':
    pass
