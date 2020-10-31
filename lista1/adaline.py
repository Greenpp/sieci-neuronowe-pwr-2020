if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista1'

from typing import Optional, Tuple

import numpy as np
from net.activations import Bipolar
from net.data_loader import DataLoader
from net.layers import FCLayer
from net.loss_functions import MSE
from net.model import Layer, Model, ModelModule
from net.trainers import SGDTrainer
from net.training_logger import TrainingLogger
from net.weights_initializers import NormalDistributionWI

from lista1.data_generator import ANDGenerator


class Adaline(Model):
    def __init__(self, theta: float, *args: Layer) -> None:
        super().__init__(*args),
        self.final_activation = Bipolar(theta)

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
        epsilon: Optional[float] = None,
    ) -> None:
        self.epsilon = 0 if epsilon is None else epsilon
        self.model = Adaline(
            theta,
            FCLayer(
                2, 1, weight_initializer=NormalDistributionWI(weight_range), bias=bias
            ),
        )

        data = ANDGenerator(bipolar=True).get_augmented()
        test_data = ANDGenerator(bipolar=True).get_all()
        training_loader = DataLoader(data, batch_size=1)
        test_loader = DataLoader(test_data, batch_size=None)

        loss = MSE()
        self.trainer = SGDTrainer(alpha, loss)
        self.trainer.set_data_loaders(training_loader, test_loader)

    def train(self, fail_after_limit: bool) -> TrainingLogger:
        self.trainer.train(
            self.model, 1000, epsilon=self.epsilon, fail_after_limit=fail_after_limit
        )

        return self.trainer.get_logger()
