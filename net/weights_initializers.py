from abc import ABC
from typing import Iterable, Tuple

import numpy as np


class WeightInitializer(ABC):
    def get_weights(self, shape: Iterable[int]):
        pass


class NormalDistributionWI(WeightInitializer):
    def __init__(self, w_range: Tuple[float, float]) -> None:
        self.w_range = w_range

    def get_weights(self, shape: Iterable[int]):
        min_w, max_w = self.w_range
        size = max_w - min_w

        weights = np.random.rand(*shape)
        shifted_weights = weights * size + min_w
        return shifted_weights


class XavierWI(NormalDistributionWI):
    def __init__(self, input_nodes: int, output_nodes: int) -> None:
        variance = np.sqrt(2 / (input_nodes + output_nodes))
        self.w_range = (-variance, variance)


class HeWI(NormalDistributionWI):
    def __init__(self, input_nodes) -> None:
        variance = np.sqrt(2 / input_nodes)
        self.w_range = (-variance, variance)
