if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista2_dod'

from lista2.mnist_loader import MNISTLoader
from net.activations import ReLU, Sigmoid, SoftmaxCE
from net.data_loader import DataLoader
from net.layers import Dropout, FCLayer
from net.loss_functions import CrossEntropy
from net.model import Model, ModelModule
from net.regularizers import get_regularizer_by_name
from net.trainers import AdamTrainer, SGDTrainer
from net.training_logger import TrainingLogger
from net.weights_initializers import RangeWI


class RegularizedPerceptron(ModelModule):
    def __init__(
        self,
        regularization: str,
        **kwargs: float,
    ) -> None:
        self.regularizer = get_regularizer_by_name(regularization)(**kwargs)

        layers = [
            FCLayer(784, 512, weight_initializer=RangeWI((-0.1, 0.1))),
            ReLU(),
            FCLayer(512, 256, weight_initializer=RangeWI((-0.1, 0.1))),
            ReLU(),
            FCLayer(256, 128, weight_initializer=RangeWI((-0.1, 0.1))),
            ReLU(),
            FCLayer(128, 10, weight_initializer=RangeWI((-0.1, 0.1))),
            SoftmaxCE(),
        ]

        self.model = Model(*layers)

        tr_data, _, te_data = MNISTLoader().get_sets()
        training_loader = DataLoader(tr_data[:10000], batch_size=1024)
        test_loader = DataLoader(te_data, batch_size=None, random=False)

        loss = CrossEntropy()
        self.trainer = SGDTrainer(0.001, loss)
        self.trainer.set_data_loaders(training_loader, test_loader)

    def train(self, verbose: bool = False) -> TrainingLogger:
        self.trainer.train(
            self.model,
            max_epochs=40,
            verbose=verbose,
            test_every_nth_batch=1,
            regularizer=self.regularizer,
        )

        return self.trainer.get_logger()


class DropPerceptron(ModelModule):
    def __init__(
        self,
        drop_rate: float,
    ) -> None:

        layers = [
            FCLayer(784, 512, weight_initializer=RangeWI((-0.1, 0.1))),
            ReLU(),
            Dropout(drop_rate),
            FCLayer(512, 256, weight_initializer=RangeWI((-0.1, 0.1))),
            ReLU(),
            Dropout(drop_rate),
            FCLayer(256, 128, weight_initializer=RangeWI((-0.1, 0.1))),
            ReLU(),
            Dropout(drop_rate),
            FCLayer(128, 10, weight_initializer=RangeWI((-0.1, 0.1))),
            SoftmaxCE(),
        ]

        self.model = Model(*layers)

        tr_data, _, te_data = MNISTLoader().get_sets()
        training_loader = DataLoader(tr_data[:10000], batch_size=1024)
        test_loader = DataLoader(te_data, batch_size=None, random=False)

        loss = CrossEntropy()
        self.trainer = AdamTrainer(0.001, loss)
        self.trainer.set_data_loaders(training_loader, test_loader)

    def train(self, verbose: bool = False) -> TrainingLogger:
        self.trainer.train(
            self.model,
            max_epochs=40,
            verbose=verbose,
            test_every_nth_batch=1,
        )

        return self.trainer.get_logger()