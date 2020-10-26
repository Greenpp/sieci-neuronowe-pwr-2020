import gzip
import pickle
import random
from typing import List, Tuple

import numpy as np


class MNISTLoader:
    def __init__(self, data_shape: Tuple[int, int] = (1, 784)) -> None:
        self.training_data = None
        self.validation_data = None
        self.test_data = None

        self._load_from_gzip()
        self._transform(data_shape)

    def _load_from_gzip(self):
        with gzip.open('./mnist.pkl.gz', 'rb') as f:
            tr, v, te = pickle.load(f, encoding='latin1')

        self.training_data = tr
        self.validation_data = v
        self.test_data = te

    def _one_hot_encode(self, pos: int) -> np.ndarray:
        encoded = np.zeros((1, 10))
        encoded[0][pos] = 1.0

        return encoded

    def _transform(self, data_shape: Tuple[int, int]):
        training_inputs = [np.reshape(d, data_shape) for d in self.training_data[0]]
        training_labels = [self._one_hot_encode(d) for d in self.training_data[1]]
        self.training_data = list(zip(training_inputs, training_labels))

        validation_inputs = [np.reshape(d, data_shape) for d in self.validation_data[0]]
        validation_labels = [self._one_hot_encode(d) for d in self.validation_data[1]]
        self.validation_data = list(zip(validation_inputs, validation_labels))

        test_inputs = [np.reshape(d, data_shape) for d in self.test_data[0]]
        test_labels = [self._one_hot_encode(d) for d in self.test_data[1]]
        self.test_data = list(zip(test_inputs, test_labels))

    def get_sets(self) -> Tuple[List, List, List]:
        return self.training_data, self.validation_data, self.test_data

    def get_random_test_example(self) -> Tuple[np.ndarray, np.ndarray]:
        return random.choice(self.test_data)


if __name__ == "__main__":
    pass
