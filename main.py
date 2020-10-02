import numpy as np

from activations import a_bipolar, a_unipolar
from data_generator import ANDGenerator
from layers import FCLayer
from model import Model


def unipolar_test() -> None:
    data_gen = ANDGenerator()
    model = Model(
        FCLayer(2, 1, a_unipolar)
    )

    data = data_gen.get_all()

    model.train(data, verbose=True)
    model.test(data)


if __name__ == "__main__":
    unipolar_test()
