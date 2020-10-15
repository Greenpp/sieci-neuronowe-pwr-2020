if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista1'

from typing import Tuple

import numpy as np
from net.activations import get_activation_by_name
from net.data_loader import DataLoader
from net.layers import FCLayer
from net.loss_functions import MSE
from net.model import Model, ModelLogger, ModelModule
from net.trainers import SGDTrainer

from lista1.data_generator import ANDGenerator


class ANDPerceptron(ModelModule):
    def __init__(
        self,
        bipolar: bool,
        theta: float,
        bias: bool,
        weight_range: Tuple[float, float],
        alpha: float,
    ) -> None:
        activation_name = 'bipolar' if bipolar else 'unipolar'
        activation = get_activation_by_name(activation_name)(theta)

        self.model = Model(FCLayer(2, 1, activation, bias, weight_range))
        self.trainer = SGDTrainer(alpha)

        data = ANDGenerator(bipolar).get_augmented()
        val_data = ANDGenerator(bipolar).get_all()
        self.training_data_loader = DataLoader(data, batch_size=1)
        self.validation_data_loader = DataLoader(val_data, batch_size=None)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.model(x)

    def train(self, fail_after_max_epochs: bool = True) -> ModelLogger:
        logger = self.model.train(
            self.training_data_loader,
            self.validation_data_loader,
            self.trainer,
            MSE(),
            max_epochs=1000,
            fail_after_max_epochs=fail_after_max_epochs,
        )

        return logger
