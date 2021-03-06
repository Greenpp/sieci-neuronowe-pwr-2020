from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from net.training_logger import TrainingLogger


class Layer(ABC):
    trainable = False

    @abstractmethod
    def __call__(self, x: np.ndarray, train: bool) -> np.ndarray:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def backward(
        self, grad: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        pass


class Model:
    """
    Sequential neural network model
    """

    def __init__(self, *args: Layer) -> None:
        self.layers = args

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.compute(x)

    def compute(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        """
        Computes network output
        """

        for layer in self.layers:
            x = layer(x, train)

        return x

    # TODO log best weights


class ModelModule(ABC):
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.model(x)

    @abstractmethod
    def train(self) -> TrainingLogger:
        pass


if __name__ == '__main__':
    pass
