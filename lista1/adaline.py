if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista1'

from net.loss_functions import MSE
from net.data_loader import DataLoader
from lista1.data_generator import ANDGenerator
from net.trainers import SGDTrainer
from typing import Tuple

import numpy as np
from net.activations import get_activation_by_name
from net.layers import FCLayer, Layer
from net.model import Model, ModelLogger, ModelModule


class Adaline(Model):
    def __init__(self, theta: float, *args: Layer) -> None:
        super().__init__(*args),
        self.final_activation = get_activation_by_name('bipolar')(theta)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        forward = self.compute(x)
        activated = self.final_activation(forward)

        return activated


class ANDAdaline(ModelModule):
    def __init__(
        self,
        theta: float,
        bias: bool,
        weight_range: Tuple[float, float],
        alpha: float,
        epsilon: float = None,
    ) -> None:
        self.epsilon = epsilon

        activation = get_activation_by_name('linear')()

        self.model = Adaline(theta, FCLayer(2, 1, activation, bias, weight_range))
        self.trainer = SGDTrainer(alpha)

        data = ANDGenerator(bipolar=True).get_augmented()
        val_data = ANDGenerator(bipolar=True).get_all()
        self.training_data_loader = DataLoader(data, batch_size=1)
        self.validation_data_loader = DataLoader(val_data, batch_size=None)

    def train(self, fail_after_max_epochs: bool = True) -> ModelLogger:
        logger = self.model.train(
            self.training_data_loader,
            self.validation_data_loader,
            self.trainer,
            MSE(),
            self.epsilon,
            max_epochs=1000,
            fail_after_max_epochs=fail_after_max_epochs,
        )

        return logger
