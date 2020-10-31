if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista1'

from typing import Tuple

import numpy as np
from net.activations import Bipolar, Unipolar
from net.data_loader import DataLoader
from net.layers import FCLayer
from net.loss_functions import MSE
from net.model import Model, ModelModule
from net.trainers import SGDTrainer
from net.training_logger import TrainingLogger
from net.weights_initializers import NormalDistributionWI

from lista1.data_generator import ANDGenerator


class ANDPerceptron(ModelModule):
    def __init__(
        self,
        weight_range: Tuple[float, float],
        alpha: float,
        bipolar: bool,
        theta: float,
        bias: bool,
    ) -> None:
        if bipolar:
            activation = Bipolar(theta)
        else:
            activation = Unipolar(theta)

        self.model = Model(
            FCLayer(
                2, 1, weight_initializer=NormalDistributionWI(weight_range), bias=bias
            ),
            activation,
        )

        tr_data = ANDGenerator(bipolar).get_augmented()
        test_data = ANDGenerator(bipolar).get_all()
        training_loader = DataLoader(tr_data, batch_size=1)
        test_loader = DataLoader(test_data, batch_size=None)

        loss = MSE()
        self.trainer = SGDTrainer(alpha, loss)
        self.trainer.set_data_loaders(training_loader, test_loader)

    def train(self, fail_after_limit: bool) -> TrainingLogger:
        self.trainer.train(
            self.model, max_epochs=1000, fail_after_limit=fail_after_limit, epsilon=0
        )

        return self.trainer.get_logger()
