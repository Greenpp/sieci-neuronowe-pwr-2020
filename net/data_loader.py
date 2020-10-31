import math
import random
from typing import Iterator, List, Optional, Tuple

import numpy as np


class DataLoader:
    """
    Data loader
    """

    def __init__(
        self,
        data: List,
        batch_size: Optional[int] = 32,
        random: bool = True,
    ) -> None:
        self.data = data
        self.batch_size = batch_size if batch_size is not None else len(data)
        self.random = random

    def _batchify(self) -> Iterator[List]:
        """
        Create batches of data
        """
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i : i + self.batch_size]

    def _randomize(self) -> None:
        """
        Shuffle data
        """
        random.shuffle(self.data)

    def _stack_batch(self, batch: List[Tuple[np.ndarray]]) -> Tuple[np.ndarray]:
        """
        Stack batch data into single tensor
        """
        input_list, output_list = zip(*batch)
        input = np.vstack(input_list)
        output = np.vstack(output_list)

        return input, output

    def load(self) -> Iterator[List]:
        """
        Load data batches as iterator
        """
        if self.random:
            self._randomize()

        for batch in self._batchify():
            stacked_batch = self._stack_batch(batch)
            yield stacked_batch

    def get_batch_num(self) -> int:
        batch_num = len(self.data) / self.batch_size
        batch_num = math.ceil(batch_num)

        return batch_num


if __name__ == "__main__":
    dl = DataLoader(np.array(range(10)))
    for d in dl.load():
        print(d)
