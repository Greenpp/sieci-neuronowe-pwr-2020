from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from net.data_loader import DataLoader
    from net.layers import Layer
    from net.loss_functions import LossFunction
    from net.trainers import Trainer


class ModelLogger:
    def __init__(self) -> None:
        self.val_errors = []
        self.test_errors = []
        self.train_errors = []
        self.failed = False
        self.weights = []
        self.biases = []

    def log_val_error(self, error: np.ndarray) -> None:
        self.val_errors.append(error)

    def log_test_error(self, error: np.ndarray) -> None:
        self.test_errors.append(error)

    def log_train_error(self, error: np.ndarray) -> None:
        self.train_errors.append(error)

    def fail(self) -> None:
        self.failed = True

    def log_weights_and_biases(
        self, weights: List[np.ndarray], biases: List[np.ndarray]
    ) -> None:
        self.weights.append(weights)
        self.biases.append(biases)

    def get_logs(self) -> dict:
        return {
            'val_errors': self.val_errors,
            'test_errors': self.test_errors,
            'train_errors': self.train_errors,
            'epochs': len(self.val_errors)
            - 1,  # first error log is before training loop
            'failed': self.failed,
            'weights': self.weights,
            'biases': self.biases,
        }


class Model:
    """
    Sequential neural network model
    """

    def __init__(self, *args: Layer) -> None:
        self.layers = args

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.compute(x)

    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Computes network output
        """

        for layer in self.layers:
            x = layer(x)

        return x

    def _stack_batch(self, batch: List[Tuple[np.ndarray]]) -> Tuple[np.ndarray]:
        """
        Stack batch data into single tensor
        """
        input_list, output_list = zip(*batch)
        input = np.vstack(input_list)
        output = np.vstack(output_list)

        return input, output

    def _log_layers(self, logger: ModelLogger) -> None:
        weights = []
        biases = []
        for l in self.layers:
            w = l.weights.copy()
            b = l.b_weights.copy() if l.bias else 0

            weights.append(w)
            biases.append(b)

        logger.log_weights_and_biases(weights, biases)

    def train(
        self,
        training_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        trainer: Trainer,
        loss_function: LossFunction,
        epsilon: float = None,
        max_epochs: int = None,
        fail_after_max_epochs=False,
        test_data_loader: DataLoader = None,
    ) -> ModelLogger:
        """
        Train model
        """
        logger = ModelLogger()

        test_data_loader = (
            test_data_loader if test_data_loader is not None else validation_data_loader
        )

        # TODO Typing.Optional for all None default arguments
        trainer.attach(self)

        val_error = self.validate(validation_data_loader, loss_function)
        test_error = self.test(test_data_loader, loss_function)
        logger.log_test_error(test_error)
        logger.log_val_error(val_error)

        epoch = 0
        # Training loop
        while (test_error > 0) if epsilon is None else (val_error > epsilon):
            epoch += 1
            epoch_loses = []
            for data_batch in training_data_loader.load():
                x, y_hat = self._stack_batch(data_batch)

                y = self.compute(x)
                loss = loss_function(y, y_hat)
                trainer.train(loss_function)

                epoch_loses.append(loss)
            train_error = np.mean(epoch_loses)
            epoch_loses.clear()

            val_error = self.validate(validation_data_loader, loss_function)
            test_error = self.test(test_data_loader, loss_function)
            # Log model state
            logger.log_train_error(train_error)
            logger.log_val_error(val_error)
            logger.log_test_error(test_error)
            self._log_layers(logger)

            if max_epochs is not None and max_epochs <= epoch:
                # Break if exceeded training epoch limit
                if fail_after_max_epochs:
                    logger.fail()
                break

        return logger

    def test(self, test_data_loader: DataLoader, loss_function: LossFunction) -> float:
        # Validation data loader should be initialized with size None and return all data at once
        val_data = next(test_data_loader.load())
        x, y_hat = self._stack_batch(val_data)

        # Call can be overloaded, ex. Adaline
        y = self(x)
        error = loss_function(y, y_hat)

        # Mean error for all outputs
        m_error = error.mean()

        return m_error

    def validate(
        self, validation_data_loader: DataLoader, loss_function: LossFunction
    ) -> float:
        # Validation data loader should be initialized with size None and return all data at once
        val_data = next(validation_data_loader.load())
        x, y_hat = self._stack_batch(val_data)

        y = self.compute(x)
        error = loss_function(y, y_hat)

        # Mean error for all outputs
        m_error = error.mean()

        return m_error


class ModelModule(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def train(self, fail_after_max_epochs: bool = True) -> ModelLogger:
        pass


if __name__ == '__main__':
    pass
