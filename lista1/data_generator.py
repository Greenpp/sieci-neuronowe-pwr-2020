from typing import List, Tuple

import numpy as np


class ANDGenerator:
    """
    AND logical gate data generator
    """

    def __init__(self, bipolar: bool = False) -> None:
        if bipolar:
            x1 = (np.array([[1.0, 1.0]]), np.array([[1.0]]))
            x2 = (np.array([[1.0, -1.0]]), np.array([[-1.0]]))
            x3 = (np.array([[-1.0, -1.0]]), np.array([[-1.0]]))
            x4 = (np.array([[-1.0, 1.0]]), np.array([[-1.0]]))
        else:
            x1 = (np.array([[1.0, 1.0]]), np.array([[1.0]]))
            x2 = (np.array([[1.0, 0.0]]), np.array([[0.0]]))
            x3 = (np.array([[0.0, 0.0]]), np.array([[0.0]]))
            x4 = (np.array([[0.0, 1.0]]), np.array([[0.0]]))

        self.data = [x1, x2, x3, x4]

    def get_all(self) -> List[Tuple[np.ndarray]]:
        return self.data

    def get_augmented(
        self, num: int = 4, include_original: bool = True
    ) -> List[Tuple[np.ndarray]]:
        augmented_data = []
        for x, y in self.data:
            for _ in range(num):
                a_x = x.copy()
                a_x[0][0] = a_x[0][0] + (np.random.rand() - 0.5) / 100
                a_x[0][1] = a_x[0][0] + (np.random.rand() - 0.5) / 100

                augmented_data.append((a_x, y))

        data = augmented_data
        if include_original:
            data += self.data

        return data


if __name__ == '__main__':
    gen = ANDGenerator()
    print(gen.get_augmented())