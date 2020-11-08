from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from net.layers import TrainableLayer

L1 = 'l1'
L2 = 'l2'
L12 = 'l12'


class Regularizer(ABC):
    @abstractmethod
    def regularize(
        self,
        layer: TrainableLayer,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class L1Regularizer(Regularizer):
    def __init__(self, lambda_: float = 0.01) -> None:
        self.lambda_ = lambda_

    def regularize(
        self,
        layer: TrainableLayer,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d_weights = d_weights + self.lambda_ * np.where(
            layer.weights >= 0,
            1.0,
            -1.0,
        )
        if layer.bias:
            d_bias = d_bias + self.lambda_ * np.where(
                layer.b_weights >= 0,
                1.0,
                -1.0,
            )

        return d_bias, d_weights


class L2Regularizer(Regularizer):
    def __init__(self, lambda_: float = 0.01) -> None:
        self.lambda_ = lambda_

    def regularize(
        self,
        layer: TrainableLayer,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d_weights = d_weights + 2 * self.lambda_ * layer.weights
        if layer.bias:
            d_bias = d_bias + 2 * self.lambda_ * layer.b_weights

        return d_bias, d_weights


class L12Regularizer(Regularizer):
    def __init__(
        self,
        lambda1: float = 0.01,
        lambda2: float = 0.01,
    ) -> None:
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def regularize(
        self,
        layer: TrainableLayer,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d_weights = (
            d_weights
            + self.lambda1
            * np.where(
                layer.weights >= 0,
                1.0,
                -1.0,
            )
            + 2 * self.lambda2 * layer.weights
        )
        if layer.bias:
            d_bias = (
                d_bias
                + self.lambda1
                * np.where(
                    layer.b_weights >= 0,
                    1.0,
                    -1.0,
                )
                + 2 * self.lambda2 * layer.b_weights
            )

        return d_bias, d_weights


REGULARIZERS = {
    L1: L1Regularizer,
    L2: L2Regularizer,
    L12: L12Regularizer,
}


def get_regularizer_by_name(name: str) -> type:
    return REGULARIZERS[name]
