from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Tuple

import numpy as np

from net.training_logger import TrainingLogger

if TYPE_CHECKING:
    from net.layers import Layer
    from net.loss_functions import LossFunction
    from net.model import Model


class Trainer(ABC):
    def __init__(
        self,
        alpha: float,
        loss_function: LossFunction,
    ) -> None:
        self.alpha = alpha
        self.loss_function = loss_function

        self.logger = TrainingLogger()

        self.layers = []

    def _attach(self, model: Model) -> None:
        self.layers = [(l, dict()) for l in reversed(model.layers)]

    def _test(self, model: Model) -> Tuple[float, float]:
        x, y = next(self.test_data_loader.load())
        y_hat = model(x)

        test_error = self.loss_function(y_hat, y)

        # TODO add case for non classification
        result_classes = y_hat.argmax(axis=1)
        label_classes = y.argmax(axis=1)
        acc = (result_classes == label_classes).mean()

        return test_error, acc

    def set_data_loaders(
        self,
        train_data_loader: Iterable,
        test_data_loader: Iterable,
        val_data_loader: Iterable = None,
    ) -> None:
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

    def train(
        self,
        model: Model,
        max_epochs=None,
        max_batches=None,
        epsilon=None,
        fail_after_max=False,
        verbose=False,
    ) -> None:
        self._attach(model)
        self._init_params()

        epoch = 0
        batch = 0
        total_batches = 0

        test_error, test_accuracy = self._test(model)
        self.logger.log_test_error(test_error)
        self.logger.log_accuracy(test_accuracy)

        epoch_batches = (
            self.train_data_loader.get_batch_num()
            if max_batches is None
            else max_batches
        )
        epoch_batches_len = len(str(epoch_batches))

        # Training loop
        is_training = True
        while is_training:
            epoch += 1
            if verbose:
                self._print_epoch(epoch)

            batch = 0
            for x, y in self.train_data_loader.load():
                total_batches += 1
                batch += 1
                if verbose:
                    self._print_left_batches(batch, epoch_batches, epoch_batches_len)

                y_hat = model(x)
                loss = self.loss_function(y_hat, y)

                grad = self.loss_function.backward()
                for layer, params in self.layers:
                    d_bias, d_weights, grad = layer.backward(grad)
                    self._update_paramas(params, d_bias, d_weights)
                    self._update_layer_weights(layer, params, d_bias, d_weights)

                # Logging errors and accuracy
                test_error, test_accuracy = self._test(model)
                self.logger.log_test_error(test_error)
                self.logger.log_accuracy(test_accuracy)
                self.logger.log_train_error(loss)

                # End conditions
                if max_batches is not None:
                    if total_batches >= max_batches:
                        is_training = False
                        if fail_after_max:
                            self.logger.log_fail()
                        break
                elif epsilon is not None:
                    if test_error <= epsilon:
                        is_training = False
                        break

            if verbose:
                # New line after line overwriting
                print('')

            if max_epochs is not None:
                if epoch >= max_epochs:
                    is_training = False
                    if fail_after_max:
                        self.logger.log_fail()

    def _print_epoch(self, epoch: int) -> None:
        print(f'Epoch {epoch}')

    def _print_left_batches(
        self, batch: int, all_batches: int, format_len: int
    ) -> None:
        print(f'\rBatch: {batch:{format_len}}/{all_batches}', end='')

    def get_logger(self) -> TrainingLogger:
        return self.logger

    def _update_paramas(
        self, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        pass

    def _init_params(self):
        pass

    @abstractmethod
    def _update_layer_weights(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        pass


class SGDTrainer(Trainer):
    def _update_layer_weights(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        layer.weights = layer.weights - self.alpha * d_weights

        if layer.bias:
            layer.b_weights = layer.b_weights - self.alpha * d_bias


class MomentumTrainer(Trainer):
    def __init__(
        self, alpha: float, loss_function: LossFunction, beta: float = 0.5
    ) -> None:
        super().__init__(alpha, loss_function)
        self.beta = beta

    def _init_params(self):
        for _, params in self.layers:
            params['prev_w_grad'] = [0]
            params['prev_b_grad'] = [0]

    def _update_paramas(
        self, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        params['prev_w_grad'].append(d_weights)
        params['prev_b_grad'].append(d_bias)

    def _update_layer_weights(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        momentum_weight = params['prev_w_grad'].pop(0)
        update_gradient = self.beta * momentum_weight + (1 - self.beta) * d_weights
        layer.weights = layer.weights - self.alpha * update_gradient

        if layer.bias:
            momentum_bias = params['prev_b_grad'].pop(0)
            update_gradient = self.beta * momentum_bias + (1 - self.beta) * d_bias
            layer.b_weights = layer.b_weights - self.alpha * update_gradient


class AdaGradTrainer(Trainer):
    def _init_params(self):
        for _, params in self.layers:
            params['w_grad_accumulator'] = 0
            params['b_grad_accumulator'] = 0

    def _update_paramas(
        self, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        params['w_grad_accumulator'] += d_weights ** 2
        params['b_grad_accumulator'] += d_bias ** 2

    def _update_layer_weights(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        weights_accumulator = params['w_grad_accumulator']
        adagrad_alpha = self.alpha / np.sqrt(weights_accumulator + 1e-9)
        layer.weights = layer.weights - adagrad_alpha * d_weights

        if layer.bias:
            bias_accumulator = params['b_grad_accumulator']
            adagrad_alpha = self.alpha / np.sqrt(bias_accumulator + 1e-9)
            layer.b_weights = layer.b_weights - adagrad_alpha * d_bias
