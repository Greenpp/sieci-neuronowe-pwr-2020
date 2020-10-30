if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista4'


from lista2.mnist_loader import MNISTLoader
from matplotlib import pyplot as plt
from net.activations import ReLU, SoftmaxCE
from net.data_loader import DataLoader
from net.layers import ConvLayer, FCLayer, Flatten, MaxPoll
from net.loss_functions import CrossEntropy
from net.model import Model
from net.trainers import AdamTrainer
from net.weights_initializers import HeWI

if __name__ == "__main__":
    data_loader = MNISTLoader(data_shape=(1, 1, 28, 28))
    tr, _, tes = data_loader.get_sets()

    train_loader = DataLoader(tr)
    test_loader = DataLoader(tes, 1000, False)

    model = Model(
        ConvLayer(1, 5, 5, padding=2, weight_initializer=HeWI(784)),
        ReLU(),
        MaxPoll(),
        Flatten(),
        FCLayer(14 * 14 * 5, 10, weight_initializer=HeWI(14 * 14 * 5)),
        SoftmaxCE(),
    )

    loss = CrossEntropy()
    trainer = AdamTrainer(0.01, loss)
    trainer.set_data_loaders(train_loader, test_loader)

    trainer.train(model, max_batches=150, verbose=True)
    logger = trainer.get_logger()

    acc = logger.get_logs()['accuracies']
    print(f'Best accuracy: {max(acc) * 100:.2f}%')
    plt.plot(acc)
    plt.show()
