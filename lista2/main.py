if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista2'


from net.data_loader import DataLoader
from lista2.mnist_loader import MNISTLoader
from .mlp import MNISTMLP
import numpy as np

if __name__ == "__main__":
    model = MNISTMLP(
        layers_shapes=[(784, 128), (128, 10)],
        activations=['sigmoid', 'softmax'],
        weight_range=(-0.5, 0.5),
        alpha=0.01,
        loss='mse',
        max_epochs=1000,
    )

    data, _, _ = MNISTLoader().get_sets()
    loader = DataLoader(data, batch_size=1)

    test_point = next(loader.load())[0][0]

    res = model(test_point)
    c = np.argmax(res)

    print(f'Out: {res}')
    print(f'Class: {c}')
