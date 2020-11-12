import numpy as np
from lista2.mnist_loader import MNISTLoader
from net.activations import SoftmaxCE, get_activation_by_name
from net.data_loader import DataLoader
from net.layers import FCLayer
from net.loss_functions import CrossEntropy
from net.model import Model, ModelModule
from net.trainers import get_trainer_by_name
from net.training_logger import TrainingLogger
from net.weights_initializers import HeWI, RangeWI, XavierWI


class OptimMNISTMLP(ModelModule):
    def __init__(
        self,
        activation_name: str,
        optimizer_name: str,
        initializer_name: str,
    ) -> None:
        tr_data, _, te_data = MNISTLoader().get_sets()
        training_loader = DataLoader(tr_data, batch_size=32)
        test_loader = DataLoader(te_data, batch_size=None, random=False)

        loss = CrossEntropy()
        self.trainer = get_trainer_by_name(optimizer_name)(
            alpha=0.01, loss_function=loss
        )
        self.trainer.set_data_loaders(training_loader, test_loader)

        hidden_size = 512
        if initializer_name == 'he':
            initializer1 = HeWI(784)
            initializer2 = HeWI(hidden_size)
        elif initializer_name == 'xavier':
            initializer1 = XavierWI(784, hidden_size)
            initializer2 = XavierWI(hidden_size, 10)
        else:
            initializer1 = RangeWI((-0.1, 0.1))
            initializer2 = RangeWI((-0.1, 0.1))

        activation = get_activation_by_name(activation_name)()

        self.model = Model(
            FCLayer(784, hidden_size, weight_initializer=initializer1),
            activation,
            FCLayer(hidden_size, 10, weight_initializer=initializer2),
            SoftmaxCE(),
        )

    def train(self, verbose: bool = False) -> TrainingLogger:
        self.trainer.train(self.model, max_epochs=5, verbose=verbose, test_every_nth_batch=32)

        return self.trainer.get_logger()
