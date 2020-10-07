from abc import ABC, abstractmethod
from net.loss_functions import LossFunction
from typing import List, Tuple

import numpy as np

from net.data_loader import DataLoader
from net.trainers import Trainer


class Layer(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, delta: np.ndarray) -> np.ndarray:
        pass


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
            x = layer.forward(x)

        return x

    def _stack_batch(batch: List[Tuple[np.ndarray]]) -> Tuple[np.ndarray]:
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
    ) -> None:
        """
        Train model
        """
        # Validation setup
        val_batch = next(validation_data_loader.load())
        val_x, val_y_hat = self._stack_batch(val_batch)
        val_error = epsilon + 1
        # Training loop
        while val_error > epsilon:
            for data_batch in training_data_loader.load():
                x, y_hat = self._stack_batch(data_batch)

                y = self.compute(x)
                error = loss_function(y, y_hat)

                trainer.train(error)

                # Validation
                val_y = self.compute(val_x)
                val_error = loss_function(val_y, val_y_hat)

    def test(self, data: List[Tuple[np.ndarray]]):
        for dp in data:
            x, y_hat = dp
            y = self.compute(x)
            print(f"{x} ==> {y} | {y_hat}")


if __name__ == "__main__":
    pass
