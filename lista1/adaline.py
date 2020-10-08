if __name__ == "__main__" and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = "lista1"

import numpy as np
from net.activations import Linear, Bipolar
from net.layers import FCLayer
from net.model import Model


class Adaline(Model):
    def __init__(self, in_: int, out: int) -> None:
        self.layers = [FCLayer(in_, out, Linear())]
        self.activation = Bipolar()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        forward = self.compute(x)

        return self.activation(forward)
