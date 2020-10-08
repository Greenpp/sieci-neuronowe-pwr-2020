if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "lista1"

from net.loss_functions import MSE
from net.data_loader import DataLoader
from net.trainers import SGDTrainer

from .data_generator import ANDGenerator
from .perceptron import Perceptron


def print_label(label: str) -> None:
    print(f'{10*"="}{label.upper()}{10*"="}')


def unipolar_test():
    print_label("unipolar")
    data_gen = ANDGenerator()
    data = data_gen.get_all()
    v_data = data_gen.get_all()
    dl = DataLoader(data, 1, False)
    vdl = DataLoader(v_data, 1, False)

    model = Perceptron(2, 1)
    trainer = SGDTrainer(0.001)

    epochs = model.train(dl, vdl, trainer, MSE(), 0)

    print(f"Done in {epochs} epochs")
    for d, _ in data:
        print(f"{d} => {model(d)}")


def bipolar_test():
    print_label("bipolar")
    data_gen = ANDGenerator(bipolar=True)
    data = data_gen.get_all()
    v_data = data_gen.get_all()
    dl = DataLoader(data, 1, False)
    vdl = DataLoader(v_data, 1, False)

    model = Perceptron(2, 1, bipolar=True)
    trainer = SGDTrainer(0.001)

    epochs = model.train(dl, vdl, trainer, MSE(), 0)

    print(f"Done in {epochs} epochs")
    for d, _ in data:
        print(f"{d} => {model(d)}")


if __name__ == "__main__":
    unipolar_test()
    bipolar_test()
