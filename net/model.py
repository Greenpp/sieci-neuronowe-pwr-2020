from __future__ import annotations
from typing import List, TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from net.data_loader import DataLoader
    from net.layers import Layer
    from net.loss_functions import LossFunction
    from net.trainers import Trainer


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

    def train(
        self,
        training_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        trainer: Trainer,
        loss_function: LossFunction,
        epsilon: float = 0.001,
    ) -> int:
        """
        Train model
        """
        trainer.attach(self)
        trainer.set_loss_function(loss_function)

        val_error = epsilon + 1
        epoch = 0
        # Training loop
        while val_error > epsilon:
            epoch += 1
            for data_batch in training_data_loader.load():
                x, y_hat = self._stack_batch(data_batch)

                y = self.compute(x)
                trainer.train(y, y_hat)

            val_error = self.validate(validation_data_loader, loss_function)

        return epoch

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


if __name__ == '__main__':
    pass
