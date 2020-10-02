import numpy as np


def a_unipolar(x: np.ndarray) -> np.ndarray:
    """
    Unipolar activation
    """
    val = x.copy()
    val[val < 0] = 0
    val[val > 0] = 1

    return val


def a_bipolar(x: np.ndarray) -> np.ndarray:
    """
    Bipolar activation
    """
    val = x.copy()
    val[val < 0] = -1
    val[val > 0] = 1

    return val


if __name__ == "__main__":
    pass
