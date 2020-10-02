from typing import List, Tuple

import numpy as np


class ANDGenerator:
    """
    NAD logical gate data generator
    """

    def get_all(self) -> List[Tuple[np.ndarray]]:
        x1 = (np.array([[1, 1]]), np.array([[1]]))
        x2 = (np.array([[1, 0]]), np.array([[0]]))
        x3 = (np.array([[0, 0]]), np.array([[0]]))
        x4 = (np.array([[0, 1]]), np.array([[0]]))

        return [x1, x2, x3, x4]
