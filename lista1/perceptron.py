if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista1'

from net.activations import Bipolar, Unipolar
from net.layers import FCLayer
from net.model import Model


class Perceptron(Model):
    def __init__(
        self, in_: int, out: int, bipolar: bool = False, theta: float = 0
    ) -> None:
        activaton = Unipolar(theta=theta) if not bipolar else Bipolar(theta=theta)
        bias = theta == 0
        self.layers = [FCLayer(in_, out, activaton, bias=bias)]
