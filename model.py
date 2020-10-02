from typing import List, Tuple
import numpy as np
from numpy.lib.function_base import select

from layers import Layer


class Model:
    """
    Neural network model
    """

    def __init__(self, *args: Layer) -> None:
        self.layers = args

    def compute(self, x: np.ndarray) -> np.ndarray:
        """
        Computes network output
        """

        for layer in self.layers:
            x = layer.forward(x)

        return x

    def train(self, data: List[Tuple[np.ndarray]], learning_rate: float = .001, verbose: bool = False) -> None:
        """
        Train model
        """
        was_error = True
        epoch = 1

        while was_error:
            if verbose:
                print(f'Epoch: {epoch}')
            epoch += 1

            was_error = False
            for dp in data:
                x, y_hat = dp
                y = self.compute(x)
                error = y - y_hat

                if error > 0:
                    was_error = True

                # TODO update to backprop
                self.layers[0].update_weight(error, learning_rate)

    def test(self, data: List[Tuple[np.ndarray]]):
        for dp in data:
            x, y_hat = dp
            y = self.compute(x)
            print(f'Input: {x}\nCorrect output: {y_hat}\nModel output: {y}')


if __name__ == "__main__":
    pass
