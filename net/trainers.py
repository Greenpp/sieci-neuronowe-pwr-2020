from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from net.layers import Layer
    from net.loss_functions import LossFunction
    from net.model import Model


class Trainer(ABC):
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def attach(self, model: Model) -> None:
        """
        Attach to models layers before training
        """
        self.layers = list(reversed(model.layers))

    def set_loss_function(self, loss_function: LossFunction) -> None:
        self.loss_function = loss_function

    @abstractmethod
    def train(self, output: np.ndarray, label: np.ndarray) -> None:
        pass


class SGDTrainer(Trainer):
    def train(self, output: np.ndarray, label: np.ndarray) -> None:
        grad = self.loss_function.backward(output, label)
        for layer in self.layers:
            d_b, d_w, grad = layer.backward(grad)
            self._update_layer_weights(layer, d_b, d_w)

    def _update_layer_weights(
        self, layer: Layer, d_b: np.ndarray, d_w: np.ndarray
    ) -> None:
        layer.weights = layer.weights - self.alpha * d_w

        if layer.bias:
            # Accumulate bias delta for batch input
            acc_d_b = d_b.sum(axis=0)
            layer.b_weights = layer.b_weights - self.alpha * acc_d_b
