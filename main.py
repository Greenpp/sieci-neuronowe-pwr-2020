import numpy as np

from activations import a_bipolar, a_unipolar
from data_generator import ANDGenerator
from layers import FCLayer
from model import Model


def print_sep() -> None:
    print(20 * '=')


def unipolar_test() -> None:
    print_sep()
    print('UNIPOLAR')
    print_sep()
    data_gen = ANDGenerator()
    model = Model(
        FCLayer(2, 1, a_unipolar)
    )

    data = data_gen.get_all()

    model.train(data)
    model.test(data)


def bipolar_test() -> None:
    print_sep()
    print('BIPOLAR')
    print_sep()
    data_gen = ANDGenerator(bipolar=True)
    model = Model(
        FCLayer(2, 1, a_bipolar)
    )

    data = data_gen.get_all()

    model.train(data)
    model.test(data)


if __name__ == "__main__":
    unipolar_test()
    bipolar_test()
