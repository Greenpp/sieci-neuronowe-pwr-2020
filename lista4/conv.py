from lista2.mnist_loader import MNISTLoader
from net.activations import ReLU, SoftmaxCE
from net.data_loader import DataLoader
from net.layers import ConvLayer, FCLayer, Flatten, MaxPoll
from net.loss_functions import CrossEntropy
from net.model import Model, ModelModule
from net.trainers import AdamTrainer
from net.training_logger import TrainingLogger
from net.weights_initializers import HeWI


class ConvMNIST(ModelModule):
    def __init__(self, kernel_size: int) -> None:
        data_loader = MNISTLoader(data_shape=(1, 1, 28, 28))
        tr, _, tes = data_loader.get_sets()
        train_loader = DataLoader(tr)
        test_loader = DataLoader(tes, 1000, False)

        loss = CrossEntropy()
        self.trainer = AdamTrainer(0.01, loss)
        self.trainer.set_data_loaders(train_loader, test_loader)

        out_channels = 1

        # TODO calculate poll and padding from kernel size

        self.model = Model(
            ConvLayer(
                1,
                out_channels,
                kernel_size,
                weight_initializer=HeWI(kernel_size * kernel_size * out_channels),
            ),
            ReLU(),
            MaxPoll(),
            Flatten(),
            FCLayer(14 * 14 * out_channels, 128),
            ReLU(),
            FCLayer(128, 10),
            SoftmaxCE(),
        )

    def train(self, verbose: bool = False) -> TrainingLogger:
        self.trainer.train(self.model, max_batches=150, verbose=verbose)

        return self.trainer.get_logger()
