if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista2'

from typing import Tuple

import numpy as np
from net.activations import SoftmaxCE, get_activation_by_name
from net.data_loader import DataLoader
from net.layers import FCLayer
from net.loss_functions import CrossEntropy
from net.model import Model, ModelModule
from net.trainers import SGDTrainer
from net.training_logger import TrainingLogger
from net.weights_initializers import NormalDistributionWI

from .mnist_loader import MNISTLoader


class MNISTMLP(ModelModule):
    def __init__(
        self,
        hidden_size: int,
        batch_size: int,
        weights_range: Tuple[float, float],
        alpha: float,
        activation_name: str,
    ) -> None:
        activation = get_activation_by_name(activation_name)()
        self.model = Model(
            FCLayer(
                784, hidden_size, weight_initializer=NormalDistributionWI(weights_range)
            ),
            activation,
            FCLayer(
                hidden_size, 10, weight_initializer=NormalDistributionWI(weights_range)
            ),
            SoftmaxCE(),
        )

        tr_data, _, te_data = MNISTLoader().get_sets()
        training_loader = DataLoader(tr_data, batch_size=batch_size)
        test_loader = DataLoader(te_data, batch_size=None, random=False)

        loss = CrossEntropy()
        self.trainer = SGDTrainer(alpha, loss)
        self.trainer.set_data_loaders(training_loader, test_loader)

    def train(self, verbose: bool = False) -> TrainingLogger:
        self.trainer.train(self.model, max_batches=150, verbose=verbose)

        return self.trainer.get_logger()
