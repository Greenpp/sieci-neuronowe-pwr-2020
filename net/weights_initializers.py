from abc import ABC
from typing import Iterable, Tuple, Type

import numpy as np
from numpy.lib.function_base import select

RANGE = 'range'
XAVIER = 'xavier'
HE = 'he'


class WeightInitializer(ABC):
    def get_weights(self, shape: Iterable[int]):
        pass


class RangeWI(WeightInitializer):
    def __init__(self, w_range: Tuple[float, float]) -> None:
        self.w_range = w_range

    def get_weights(self, shape: Iterable[int]):
        min_w, max_w = self.w_range
        size = max_w - min_w

        weights = np.random.rand(*shape)
        shifted_weights = weights * size + min_w
        return shifted_weights


class NormalDistributionWI(WeightInitializer):
    def __init__(self, exp: float, var: float) -> None:
        self.exp = exp
        self.dev = np.sqrt(var)

    def get_weights(self, shape: Iterable[int]):
        return np.random.normal(loc=self.exp, scale=self.dev, size=shape)


class XavierWI(NormalDistributionWI):
    def __init__(self, input_nodes: int, output_nodes: int) -> None:
        variance = 2 / (input_nodes + output_nodes)
        super().__init__(0, variance)


class HeWI(NormalDistributionWI):
    def __init__(self, input_nodes) -> None:
        variance = 2 / input_nodes
        super().__init__(0, variance)


INITIALIZERS = {RANGE: RangeWI, XAVIER: XavierWI, HE: HeWI}


def get_initializer_by_name(name: str) -> Type[WeightInitializer]:
    return INITIALIZERS[name]
