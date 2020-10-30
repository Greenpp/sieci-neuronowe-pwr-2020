from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from net.model import Layer
from net.utils import col2im_indices, im2col_indices
from net.weights_initializers import NormalDistributionWI

if TYPE_CHECKING:
    from net.weights_initializers import WeightInitializer


class TrainableLayer(Layer):
    trainable = True


FC = 'fc'
CONV = 'conv'
MAXPOLL = 'maxpoll'
FLATTEN = 'flatten'


class FCLayer(TrainableLayer):
    """
    Fully connected layer
    """

    def __init__(
        self,
        in_: int = 1,
        out: int = 1,
        bias: bool = True,
        weight_initializer: WeightInitializer = NormalDistributionWI((-0.5, 0.5)),
    ) -> None:
        self.weights = weight_initializer.get_weights((in_, out))
        self.bias = bias
        if bias:
            self.b_weights = weight_initializer.get_weights((1, out))

        self.input_signal = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute forward pass
        """
        self.input_signal = x

        f_x = x @ self.weights
        if self.bias:
            f_x = f_x + self.b_weights

        return f_x

    def __str__(self) -> str:
        return FC

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Weights delta equal to incoming gradint * activaton derivative * previous layer
        d_w = self.input_signal.T @ grad

        # Gradient for next layer scaled with weights
        new_grad = grad @ self.weights.T

        # Accumulate bias delta for batch input
        acc_d_b = grad.sum(axis=0)

        return acc_d_b, d_w, new_grad


class ConvLayer(TrainableLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        stride: int = 1,
        padding: int = 1,
        weight_initializer: WeightInitializer = NormalDistributionWI((-0.5, 0.5)),
    ) -> None:
        kernel_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weights = weight_initializer.get_weights(kernel_shape)
        self.bias = bias
        if bias:
            self.b_weights = weight_initializer.get_weights((out_channels, 1))

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
            f_x = f_x + self.b_weights
        # Reshape to initial form
        f_x = f_x.reshape(self.filters, x_height_out, x_width_out, batch_size)
        f_x = f_x.transpose(3, 0, 1, 2)

        return f_x

    def __str__(self) -> str:
        return CONV

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO what is going on ...
        d_b = np.sum(grad, axis=(0, 2, 3))
        d_b = d_b.reshape(self.filters, -1)

        grad_col = grad.transpose(1, 2, 3, 0).reshape(self.filters, -1)
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


class MaxPoll(Layer):
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


class Flatten(Layer):
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


LAYERS = {FC: FCLayer, CONV: ConvLayer, MAXPOLL: MaxPoll, FLATTEN: Flatten}


def get_layer_by_name(name: str) -> type:
    return LAYERS[name]


if __name__ == '__main__':
    pass
