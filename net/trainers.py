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
        loss_function: LossFunction,
    ) -> None:
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
                    if d_weights is not None:  # For poll and flatten layers
                        self._update_paramas(layer, params, d_bias, d_weights)
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

        self._finish_training()

    def _print_epoch(self, epoch: int) -> None:
        print(f'Epoch {epoch}')

    def _print_left_batches(
        self, batch: int, all_batches: int, format_len: int
    ) -> None:
        print(f'\rBatch: {batch:{format_len}}/{all_batches}', end='')

    def get_logger(self) -> TrainingLogger:
        return self.logger

    def _update_paramas(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        pass

    def _init_params(self) -> None:
        pass

    def _finish_training(self) -> None:
        pass

    @abstractmethod
    def _update_layer_weights(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        pass


class SGDTrainer(Trainer):
    def __init__(self, alpha: float, loss_function: LossFunction) -> None:
        super().__init__(loss_function)
        self.alpha = alpha

    def _update_layer_weights(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        layer.weights = layer.weights - self.alpha * d_weights

        if layer.bias:
            layer.b_weights = layer.b_weights - self.alpha * d_bias


class MomentumTrainer(Trainer):
    def __init__(
        self, alpha: float, loss_function: LossFunction, beta: float = 0.9
    ) -> None:
        super().__init__(loss_function)
        self.alpha = alpha
        self.beta = beta

    def _init_params(self) -> None:
        for _, params in self.layers:
            params['prev_w_grad'] = [0, 0]
            params['prev_b_grad'] = [0, 0]

    def _update_paramas(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        momentum_weight = params['prev_w_grad'].pop(0)
        update_gradient = self.beta * momentum_weight + (1 - self.beta) * d_weights
        params['prev_w_grad'].append(update_gradient)

        if layer.bias:
            momentum_bias = params['prev_b_grad'].pop(0)
            update_gradient = self.beta * momentum_bias + (1 - self.beta) * d_bias
            params['prev_b_grad'].append(update_gradient)

    def _update_layer_weights(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        update_gradient = params['prev_w_grad'][1]
        layer.weights = layer.weights - self.alpha * update_gradient

        if layer.bias:
            update_gradient = params['prev_b_grad'][1]
            layer.b_weights = layer.b_weights - self.alpha * update_gradient


class NesterovTrainer(Trainer):
    def __init__(
        self, alpha: float, loss_function: LossFunction, beta: float = 0.9
    ) -> None:
        super().__init__(loss_function)
        self.alpha = alpha
        self.beta = beta

    def _init_params(self) -> None:
        for layer, params in self.layers:
            params['prev_w_grad'] = [0, 0]
            params['prev_b_grad'] = [0, 0]

            params['w_pre_jump'] = layer.weights
            params['b_pre_jump'] = layer.b_weights

    def _update_paramas(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        momentum_weight = params['prev_w_grad'].pop(0)
        update_gradient = self.beta * momentum_weight + (1 - self.beta) * d_weights
        params['prev_w_grad'].append(update_gradient)

        if layer.bias:
            momentum_bias = params['prev_b_grad'].pop(0)
            update_gradient = self.beta * momentum_bias + (1 - self.beta) * d_bias
            params['prev_b_grad'].append(update_gradient)

    def _update_layer_weights(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        update_gradient = self.alpha * params['prev_w_grad'][1]
        pre_jump_w = params['w_pre_jump']

        # Momentum step and save position
        layer.weights = pre_jump_w - update_gradient
        params['w_pre_jump'] = layer.weights

        # Make jump for next gradient calculation
        layer.weights = layer.weights - update_gradient

        if layer.bias:
            update_gradient = self.alpha * params['prev_b_grad'][1]
            pre_jump_b = params['b_pre_jump']

            # Momentum step and save position
            layer.b_weights = pre_jump_b - update_gradient
            params['b_pre_jump'] = layer.b_weights

            # Make jump for next gradient calculation
            layer.b_weights = layer.b_weights - update_gradient

    def _finish_training(self) -> None:
        # Exit training with pre jump weights
        for layer, params in self.layers:
            layer.weights = params['w_pre_jump']

            if layer.bias:
                layer.b_weights = params['b_pre_jump']


class AdaGradTrainer(Trainer):
    def __init__(self, alpha: float, loss_function: LossFunction) -> None:
        super().__init__(loss_function)
        self.alpha = alpha

    def _init_params(self) -> None:
        for _, params in self.layers:
            params['w_grad_accumulator'] = 0
            params['b_grad_accumulator'] = 0

    def _update_paramas(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        params['w_grad_accumulator'] += d_weights ** 2
        if layer.bias:
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


class AdaDeltaTrainer(Trainer):
    def __init__(
        self, alpha: float, loss_function: LossFunction, gamma: float = 0.9
    ) -> None:
        super().__init__(alpha, loss_function)
        self.gamma = gamma

    def _init_params(self) -> None:
        for _, params in self.layers:
            params['delta_w_running_avg'] = 0
            params['delta_b_running_avg'] = 0

            params['grad_w_running_avg'] = 0
            params['grad_b_running_avg'] = 0

    def _update_paramas(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        params['delta_w_running_avg'] = self.gamma * params['delta_w_running_avg'] + (
            1 - self.gamma
        ) * (d_weights ** 2)

        if layer.bias:
            params['delta_b_running_avg'] = self.gamma * params[
                'delta_b_running_avg'
            ] + (1 - self.gamma) * (d_bias ** 2)

    def _update_layer_weights(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        update_gradient = (
            np.sqrt(params['grad_w_running_avg'] + 1e-9)
            / np.sqrt(params['delta_w_running_avg'] + 1e-9)
        ) * d_weights
        layer.weights = layer.weights - update_gradient

        params['grad_w_running_avg'] = self.gamma * params['grad_w_running_avg'] + (
            1 - self.gamma
        ) * (update_gradient ** 2)

        if layer.bias:
            update_gradient = (
                np.sqrt(params['grad_b_running_avg'] + 1e-9)
                / np.sqrt(params['delta_b_running_avg'] + 1e-9)
            ) * d_bias
            layer.b_weights = layer.b_weights - update_gradient

            params['grad_b_running_avg'] = self.gamma * params['grad_b_running_avg'] + (
                1 - self.gamma
            ) * (update_gradient ** 2)


class AdamTrainer(Trainer):
    def __init__(
        self,
        alpha: float,
        loss_function: LossFunction,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> None:
        super().__init__(loss_function)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2

    def _init_params(self) -> None:
        for _, params in self.layers:
            params['w_m_momentum'] = 0
            params['w_v_momentum'] = 0

            params['b_m_momentum'] = 0
            params['b_v_momentum'] = 0

            params['beta1_cor'] = 1
            params['beta2_cor'] = 1

    def _update_paramas(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        params['w_m_momentum'] = (
            self.beta1 * params['w_m_momentum'] + (1 - self.beta1) * d_weights
        )
        params['w_v_momentum'] = self.beta2 * params['w_v_momentum'] + (
            1 - self.beta2
        ) * (d_weights ** 2)

        if layer.bias:
            params['b_m_momentum'] = (
                self.beta1 * params['b_m_momentum'] + (1 - self.beta1) * d_bias
            )
            params['b_v_momentum'] = self.beta2 * params['b_v_momentum'] + (
                1 - self.beta2
            ) * (d_bias ** 2)

        params['beta1_cor'] *= self.beta1
        params['beta2_cor'] *= self.beta2

    def _update_layer_weights(
        self, layer: Layer, params: dict, d_bias: np.ndarray, d_weights: np.ndarray
    ) -> None:
        # Correction calculation
        m_momentum = params['w_m_momentum'] / (1 - params['beta1_cor'])
        v_momentum = params['w_v_momentum'] / (1 - params['beta2_cor'])

        layer.weights = (
            layer.weights - (self.alpha / (np.sqrt(v_momentum) + 1e-8)) * m_momentum
        )

        if layer.bias:
            # Correction calculation
            m_momentum = params['b_m_momentum'] / (1 - params['beta1_cor'])
            v_momentum = params['b_v_momentum'] / (1 - params['beta2_cor'])

            layer.b_weights = (
                layer.b_weights
                - (self.alpha / (np.sqrt(v_momentum) + 1e-8)) * m_momentum
            )
