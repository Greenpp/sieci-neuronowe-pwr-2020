if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista2'

from typing import List, Tuple

import numpy as np
from net.activations import get_activation_by_name
from net.data_loader import DataLoader
from net.layers import FCLayer
from net.loss_functions import get_loss_by_name
from net.model import Model, ModelLogger, ModelModule
from net.trainers import SGDTrainer

from .mnist_loader import MNISTLoader


class MNISTMLP(ModelModule):
    def __init__(
        self,
        layers_shapes: List[Tuple[int, int]],
        activations: List[str],
        weight_range: Tuple[float, float],
        alpha: float,
        loss: str,
        batch_size: int,
        epsilon: float = None,
        max_epochs: int = None,
    ) -> None:
        self.epsilon = epsilon
        self.max_epochs = max_epochs

        layers = []
        for (in_, out), act_name in zip(layers_shapes, activations):
            activation = get_activation_by_name(act_name)()
            l = FCLayer(in_, out, activation, weight_range=weight_range)
            layers.append(l)

        self.model = Model(*layers)
        self.trainer = SGDTrainer(alpha)
        self.loss_function = get_loss_by_name(loss)()

        tr_data, v_data, te_data = MNISTLoader().get_sets()

        self.training_data_loader = DataLoader(tr_data, batch_size=batch_size)
        self.validation_data_loader = DataLoader(v_data, batch_size=None, random=False)
        self.test_data_loader = DataLoader(te_data, batch_size=None, random=False)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.model(x)

    def train(self, fail_after_max_epochs: bool = False) -> ModelLogger:
        logger = self.model.train(
            self.training_data_loader,
            self.validation_data_loader,
            self.trainer,
            self.loss_function,
            epsilon=self.epsilon,
            max_epochs=self.max_epochs,
            fail_after_max_epochs=fail_after_max_epochs,
            test_data_loader=self.test_data_loader,
        )

        return logger
