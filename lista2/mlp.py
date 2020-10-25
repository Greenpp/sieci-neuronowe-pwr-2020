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
from net.model import Model, ModelModule
from net.trainers import SGDTrainer
from net.training_logger import TrainingLogger

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
        max_batches: int = None,
    ) -> None:
        self.epsilon = epsilon
        self.max_epochs = max_epochs
        self.max_batches = max_batches

        layers = []
        for (in_, out), act_name in zip(layers_shapes, activations):
            activation = get_activation_by_name(act_name)()
            l = FCLayer(in_, out, activation, weight_range=weight_range)
            layers.append(l)
        self.model = Model(*layers)

        tr_data, _, te_data = MNISTLoader().get_sets()
        training_data_loader = DataLoader(tr_data, batch_size=batch_size)
        test_data_loader = DataLoader(te_data, batch_size=None, random=False)

        loss_function = get_loss_by_name(loss)()
        self.trainer = SGDTrainer(alpha, loss_function)
        self.trainer.set_data_loaders(training_data_loader, test_data_loader)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.model(x)

    def train(
        self, fail_after_max_epochs: bool = False, verbose: bool = False
    ) -> TrainingLogger:
        trainer = self.trainer
        model = self.model

        trainer.train(
            model,
            max_epochs=self.max_epochs,
            max_batches=self.max_batches,
            epsilon=self.epsilon,
            fail_after_max=fail_after_max_epochs,
            verbose=verbose,
        )

        logger = trainer.get_logger()

        return logger
