from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, Type

import numpy as np

from net.training_logger import TrainingLogger

if TYPE_CHECKING:
    from net.data_loader import DataLoader
    from net.layers import TrainableLayer
    from net.loss_functions import LossFunction
    from net.model import Model
    from net.regularizers import Regularizer


SGD = 'sgd'
MOMENTUM = 'momentum'
NESTEROV = 'nesterov'
ADAGRAD = 'adagrad'
ADADELTA = 'adadelta'
ADAM = 'adam'


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
        ys = []
        y_hats = []
        for x, y in self.test_data_loader.load():
            ys.append(y)

            y_hat = model(x)
            y_hats.append(y_hat)

        y = np.vstack(ys)
        y_hat = np.vstack(y_hats)

        test_error = self.loss_function(y_hat, y)

        if y.shape[1] > 1:
            result_classes = y_hat.argmax(axis=1)
            label_classes = y.argmax(axis=1)
            acc = (result_classes == label_classes).mean()
        else:
            acc = None

        return test_error, acc

    def set_data_loaders(
        self,
        train_data_loader: DataLoader,
        test_data_loader: DataLoader,
        val_data_loader: Optional[DataLoader] = None,
    ) -> None:
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

    def train(
        self,
        model: Model,
        max_epochs: Optional[int] = None,
        max_batches: Optional[int] = None,
        epsilon: Optional[float] = None,
        fail_after_limit: bool = False,
        verbose: bool = False,
        test_every_nth_batch: int = 1,
        regularizer: Optional[Regularizer] = None,
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
                    self._print_left_batches_and_accuracy(
                        batch, epoch_batches, epoch_batches_len, test_accuracy
                    )

                y_hat = model.compute(x, train=True)
                loss = self.loss_function(y_hat, y)

                grad = self.loss_function.backward()
                for layer, params in self.layers:
                    d_bias, d_weights, grad = layer.backward(grad)
                    if layer.trainable:
                        if regularizer is not None:
                            d_bias, d_weights = regularizer.regularize(
                                layer, d_bias, d_weights
                            )
                        self._update_paramas(layer, params, d_bias, d_weights)
                        self._update_layer_weights(layer, params, d_bias, d_weights)

                # Logging errors and accuracy
                if batch % test_every_nth_batch == 0:
                    test_error, test_accuracy = self._test(model)
                    self.logger.log_test_error(test_error)
                    self.logger.log_accuracy(test_accuracy)
                    self.logger.log_train_error(loss)
                self.logger.log_batch()

                # End conditions
                if epsilon is not None:
                    if test_error <= epsilon:
                        is_training = False
                        break
                if max_batches is not None:
                    if total_batches >= max_batches:
                        is_training = False
                        if fail_after_limit:
                            self.logger.log_fail()
                        break

            if verbose:
                # New line after line overwriting
                print('')

            if max_epochs is not None:
                if epoch >= max_epochs:
                    is_training = False
                    if fail_after_limit:
                        self.logger.log_fail()

            self.logger.log_epoch()

        test_error, test_accuracy = self._test(model)
        self.logger.log_test_error(test_error)
        self.logger.log_accuracy(test_accuracy)
        self._finish_training()

    def test_gradient(
        self,
        model: Model,
        epsilon: float = 1e-7,
    ) -> None:
        self._attach(model)
        self._init_params()

        for x, y in self.test_data_loader.load():
            pass

        x = x[0:1, :]
        y = y[0:1, :]

        # Analytic gradients
        y_hat_a = model.compute(x)
        loss_a = self.loss_function(y_hat_a, y)

        deltas = []

        grad = self.loss_function.backward()
        for layer, _ in self.layers:
            d_bias, d_weights, grad = layer.backward(grad)
            if layer.trainable:
                delta = d_weights.reshape((-1, 1))
                deltas.append(delta)
                if layer.bias:
                    delta_b = d_bias.reshape((-1, 1))
                    deltas.append(delta_b)
        grad_analytic = np.concatenate(deltas, axis=0)

        # Numeric gradient

        grads = []
        for layer, _ in self.layers:
            if layer.trainable:
                weights = layer.weights
                w_shape = weights.shape
                weights = weights.reshape((-1, 1))
                for i in range(weights.shape[0]):
                    weights[i] = weights[i] + epsilon

                    layer.weights = weights.reshape(w_shape)
                    y_hat = model(x)
                    loss_n = self.loss_function(y_hat, y)
                    grad_n = (loss_n - loss_a) / epsilon
                    grads.append(grad_n)
                    weights[i] = weights[i] - epsilon
                layer.weights = weights.reshape(w_shape)

                if layer.bias:
                    b_weights = layer.b_weights
                    b_shape = b_weights.shape
                    b_weights = b_weights.reshape((-1, 1))
                    for i in range(b_weights.shape[0]):
                        b_weights[i] = b_weights[i] + epsilon

                        layer.b_weights = b_weights.reshape(b_shape)
                        y_hat = model(x)
                        loss_n = self.loss_function(y_hat, y)
                        grad_n = (loss_n - loss_a) / epsilon
                        grads.append(grad_n)
                        b_weights[i] = b_weights[i] - epsilon
                    layer.b_weights = b_weights.reshape(b_shape)
        grads = np.array(grads)
        grad_numeric = grads.reshape((-1, 1))

        num = np.linalg.norm(grad_numeric - grad_analytic, ord=2)
        den = np.linalg.norm(grad_numeric, ord=2) + np.linalg.norm(grad_analytic, ord=2)
        diff = num / den
        if diff < epsilon:
            print(f'Correct gradient | Diff: {diff}')
        else:
            print(f'Wrong gradient | Diff: {diff}')

    def _print_epoch(self, epoch: int) -> None:
        print(f'Epoch {epoch}')

    def _print_left_batches_and_accuracy(
        self, batch: int, all_batches: int, format_len: int, acc: float
    ) -> None:
        print(
            f'\rBatch: {batch:{format_len}}/{all_batches} | Accuracy: {acc * 100:5.2f}%',
            end='',
        )

    def get_logger(self) -> TrainingLogger:
        return self.logger

    def _update_paramas(
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
    ) -> None:
        pass

    def _init_params(self) -> None:
        pass

    def _finish_training(self) -> None:
        pass

    @abstractmethod
    def _update_layer_weights(
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
    ) -> None:
        pass


class SGDTrainer(Trainer):
    def __init__(self, alpha: float, loss_function: LossFunction) -> None:
        super().__init__(loss_function)
        self.alpha = alpha

    def _update_layer_weights(
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
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
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
    ) -> None:
        momentum_weight = params['prev_w_grad'].pop(0)
        update_gradient = self.beta * momentum_weight + (1 - self.beta) * d_weights
        params['prev_w_grad'].append(update_gradient)

        if layer.bias:
            momentum_bias = params['prev_b_grad'].pop(0)
            update_gradient = self.beta * momentum_bias + (1 - self.beta) * d_bias
            params['prev_b_grad'].append(update_gradient)

    def _update_layer_weights(
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
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

            if layer.trainable:
                params['w_pre_jump'] = layer.weights
                params['b_pre_jump'] = layer.b_weights

    def _update_paramas(
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
    ) -> None:
        momentum_weight = params['prev_w_grad'].pop(0)
        update_gradient = self.beta * momentum_weight + (1 - self.beta) * d_weights
        params['prev_w_grad'].append(update_gradient)

        if layer.bias:
            momentum_bias = params['prev_b_grad'].pop(0)
            update_gradient = self.beta * momentum_bias + (1 - self.beta) * d_bias
            params['prev_b_grad'].append(update_gradient)

    def _update_layer_weights(
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
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
            if layer.trainable:
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
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
    ) -> None:
        params['w_grad_accumulator'] += d_weights ** 2
        if layer.bias:
            params['b_grad_accumulator'] += d_bias ** 2

    def _update_layer_weights(
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
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
        super().__init__(loss_function)
        self.gamma = gamma

    def _init_params(self) -> None:
        for _, params in self.layers:
            params['delta_w_running_avg'] = 0
            params['delta_b_running_avg'] = 0

            params['grad_w_running_avg'] = 0
            params['grad_b_running_avg'] = 0

    def _update_paramas(
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
    ) -> None:
        params['delta_w_running_avg'] = self.gamma * params['delta_w_running_avg'] + (
            1 - self.gamma
        ) * (d_weights ** 2)

        if layer.bias:
            params['delta_b_running_avg'] = self.gamma * params[
                'delta_b_running_avg'
            ] + (1 - self.gamma) * (d_bias ** 2)

    def _update_layer_weights(
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
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
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
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
        self,
        layer: TrainableLayer,
        params: dict,
        d_bias: np.ndarray,
        d_weights: np.ndarray,
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


TRAINERS = {
    SGD: SGDTrainer,
    MOMENTUM: MomentumTrainer,
    NESTEROV: NesterovTrainer,
    ADAGRAD: AdaGradTrainer,
    ADADELTA: AdaDeltaTrainer,
    ADAM: AdamTrainer,
}


def get_trainer_by_name(name: str) -> Type[Trainer]:
    return TRAINERS[name]
