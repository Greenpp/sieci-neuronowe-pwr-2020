from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Type

import numpy as np

from net.model import Layer
from net.utils import col2im_indices, im2col_indices
from net.weights_initializers import RangeWI

if TYPE_CHECKING:
    from net.weights_initializers import WeightInitializer


class TrainableLayer(Layer):
    trainable = True


FC = 'fc'
CONV = 'conv'
MAXPOLL = 'maxpoll'
FLATTEN = 'flatten'
DROP = 'drop'


class FCLayer(TrainableLayer):
    """
    Fully connected layer
    """

    def __init__(
        self,
        in_: int = 1,
        out: int = 1,
        bias: bool = True,
        weight_initializer: WeightInitializer = RangeWI((-0.5, 0.5)),
    ) -> None:
        self.weights = weight_initializer.get_weights((in_, out))
        self.bias = bias
        if bias:
            self.b_weights = weight_initializer.get_weights((1, out))

        self.input_signal = None

    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
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
        d_w = d_w / grad.shape[0]

        # Gradient for next layer scaled with weights
        new_grad = grad @ self.weights.T

        # Accumulate bias delta for batch input
        acc_d_b = grad.mean(axis=0)

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
        weight_initializer: WeightInitializer = RangeWI((-0.5, 0.5)),
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

    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
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
        # (F_size x filter_locations * batch)
        # F_size = F_h * F_w * in_channels
        # Columns - filter_locations
        x_col = im2col_indices(
            x, self.kernel_size, self.kernel_size, self.padding, self.stride
        )

        # Kernel preparation for convolution in columns
        # (F_num x channels_in x F_h x F_w) -> (F_num x F_size)
        kernel_col = self.weights.reshape(self.filters, -1)

        # Cache
        self.input_signal_col = x_col
        self.input_signal = x

        # (F_num x F_size) @ (F_size x filter_locations * batch) -> 
        # (F_num x filter_locations * batch)
        f_x = kernel_col @ x_col
        if self.bias:
            f_x = f_x + self.b_weights
        # Reshape to initial form 
        # (F_num x filter_locations * batch) -> (F_num x out_h x out_w x batch)
        f_x = f_x.reshape(self.filters, x_height_out, x_width_out, batch_size)
        # (F_num x out_h x out_w x batch) -> (batch x F_num x out_h x out_w)
        f_x = f_x.transpose(3, 0, 1, 2)

        return f_x

    def __str__(self) -> str:
        return CONV

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # grad (batch x F_num x out_h x out_w)
        # Mean over batches for bias
        d_b = np.mean(grad, axis=(0, 2, 3))
        d_b = d_b.reshape(self.filters, -1)

        # grad (batch x F_num x out_h x out_w)
        # (batch x F_num x out_h x out_w) -> (F_num x filter_locations * batch)
        grad_col = grad.transpose(1, 2, 3, 0).reshape(self.filters, -1)

        # (F_num x filter_locations * batch) @ (batch * filter_locations x F_size)
        # F_num x F_size
        d_w = grad_col @ self.input_signal_col.T

        # Mean over batch
        d_w = d_w / grad.shape[0]

        # Reshape to weights 
        # (F_num x F_size) -> (F_num x channels_in x F_h x F_w)
        d_w = d_w.reshape(self.weights.shape)

        # (F_num x channels_in x F_h x F_w) -> (F_num x F_size)
        w_col = self.weights.reshape(self.filters, -1)

        # (F_size x F_num) @ (F_num x filter_locations * batch)
        # (F_size x filter_locations * batch)
        # Same as after im2col
        new_grad_col = w_col.T @ grad_col

        # Reverse im2col
        # (F_size x filter_locations * batch) -> (batch x in_channels x in_h x in_w)
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

    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:

        batch_size, channels, x_height, x_width = x.shape
        out_h = (x_height - self.size + 2 * self.padding) / self.stride + 1
        out_w = (x_width - self.size + 2 * self.padding) / self.stride + 1

        if not out_h.is_integer() or not out_w.is_integer():
            raise Exception('Wrong poll shape')

        out_h = int(out_h)
        out_w = int(out_w)

        # Stretch channels on separate columns
        x_reshaped = x.reshape(batch_size * channels, 1, x_height, x_width)
        

        # Transform into columns
        # (P_size x filter_locations * batch * in_channels)
        # P_size = P_h * P_w
        # Columns - poll_locations
        x_col = im2col_indices(
            x_reshaped, self.size, self.size, self.padding, self.stride
        )

        # Find max values idx
        max_idx = x_col.argmax(axis=0)

        # Cache
        self.input_signal_col = x_col
        self.input_signal = x
        self.max_idx = max_idx

        # Select only max values
        # max_id on pos n | n (0, max_id_num)
        out = x_col[max_idx, range(max_idx.size)]

        # Reshape back
        out = out.reshape(out_h, out_w, batch_size, channels)

        # (batch, channels, out_h, out_w)
        out = out.transpose(2, 3, 0, 1)

        return out

    def __str__(self) -> str:
        return MAXPOLL

    def backward(self, grad: np.ndarray) -> Tuple[None, None, np.ndarray]:
        batch_size, channels, x_height, x_width = self.input_signal.shape
        
        # (P_size x filter_locations * batch * in_channels)
        new_grad_col = np.zeros_like(self.input_signal_col)

        # grad (batch x channels x out_h x out_w)
        # (out_h x out_w x batch x channels) -> flat
        flat_grad = grad.transpose(2, 3, 0, 1).ravel()

        # Replace selected in forward values with gradient
        new_grad_col[self.max_idx, range(self.max_idx.size)] = flat_grad

        # Transform columns into matrix
        new_grad = col2im_indices(
            new_grad_col,
            (batch_size * channels, 1, x_height, x_width),
            self.size,
            self.size,
            self.padding,
            self.stride,
        )

        # Extract channels stretched in input_signal_col
        new_grad = new_grad.reshape(self.input_signal.shape)

        return None, None, new_grad


class Flatten(Layer):
    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
        self.input_shape = x.shape
        batch_size = x.shape[0]

        return x.ravel().reshape(batch_size, -1)

    def __str__(self) -> str:
        return FLATTEN

    def backward(self, grad: np.ndarray) -> Tuple[None, None, np.ndarray]:
        new_grad = grad.reshape(self.input_shape)

        return None, None, new_grad


class Dropout(Layer):
    def __init__(self, rate: float) -> None:
        self.drop_rate = rate

    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
        if train:
            signal_shape = x.shape
            self.mask = np.where(np.random.rand(*signal_shape) < self.drop_rate, 0, 1)
            x = x * self.mask

        return x

    def __str__(self) -> str:
        return DROP

    def backward(self, grad: np.ndarray) -> Tuple[None, None, np.ndarray]:
        new_grad = grad * self.mask
        return None, None, new_grad


LAYERS = {
    FC: FCLayer,
    CONV: ConvLayer,
    MAXPOLL: MaxPoll,
    FLATTEN: Flatten,
    DROP: Dropout,
}


def get_layer_by_name(name: str) -> Type[Layer]:
    return LAYERS[name]


if __name__ == '__main__':
    pass
