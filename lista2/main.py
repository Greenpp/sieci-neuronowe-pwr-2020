if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista2'


from lista2.mnist_loader import MNISTLoader
from .mlp import MNISTMLP
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    model = MNISTMLP(
        layers_shapes=[(784, 128), (128, 10)],
        activations=['relu', 'softmax_ce'],
        weight_range=(-0.5, 0.5),
        alpha=0.01,
        loss='cross-entropy',
        batch_size=32,
        max_batches=200,
    )

    logger = model.train(verbose=True)
    logs = logger.get_logs()

    mnist_data = MNISTLoader()
    for _ in range(10):
        x, y_hat = mnist_data.get_random_test_example()
        y = model(x)

        print(f'{np.argmax(y)} || {np.argmax(y_hat)}')

    acc = logs['accuracies']
    print(f'Best accuracy: {round(max(acc) * 100, 2)}%')
    plt.plot(acc)
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.show()
