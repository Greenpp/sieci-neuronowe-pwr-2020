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

    @abstractmethod
    def _update_layer_weights(
        self, layer: Layer, d_b: np.ndarray, d_w: np.ndarray
    ) -> None:
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
            layer.b_weights = layer.b_weights - self.alpha * d_b


class MomentumTrainer(Trainer):
    def __init__(self, alpha: float, beta: float = 0.5) -> None:
        super().__init__(alpha)
        self.beta = beta

    def attach(self, model: Model) -> None:
        self.layers = [
            {
                'layer': l,
                'prev_w_grad': np.zeros_like(l.weights),
                'prev_b_grad': np.zeros_like(l.b_weights) if l.bias else None,
            }
            for l in reversed(model.layers)
        ]

    def train(self, output: np.ndarray, label: np.ndarray) -> None:
        grad = self.loss_function.backward(output, label)
        for layer_dict in self.layers:
            layer = layer_dict['layer']

            b_momentum = layer_dict['prev_b_grad']
            w_momentum = layer_dict['prev_w_grad']

            d_b, d_w, grad = layer.backward(grad)
            # Update momentum gradients for next pass
            layer_dict['prev_b_grad'] = d_b
            layer_dict['prev_w_grad'] = d_w

            self._update_layer_weights(layer, d_b, d_w, b_momentum, w_momentum)

    def _update_layer_weights(
        self,
        layer: Layer,
        d_b: np.ndarray,
        d_w: np.ndarray,
        b_momentum: np.ndarray,
        w_momentum: np.ndarray,
    ) -> None:
        update_gradient = self.beta * w_momentum + (1 - self.beta) * d_w
        layer.weights = layer.weights - self.alpha * update_gradient

        if layer.bias:
            b_update_gradient = self.beta * b_momentum + (1 - self.beta) * d_b
            layer.b_weights = layer.b_weights - self.alpha * b_update_gradient