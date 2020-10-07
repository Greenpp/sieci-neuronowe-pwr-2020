from typing import Iterator, List, Tuple
import numpy as np
import random


class DataLoader:
    """
    Data loader
    """

    def __init__(self, data: List, batch_size: int = 32, infinite: bool = True, random: bool = True) -> None:
        self.data = data
        self.batch_size = batch_size
        self.infinite = infinite
        self.random = random

    def _batchify(self) -> Iterator[List]:
        """
        Create batches of data
        """
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i: i+self.batch_size]

    def _randomize(self) -> None:
        """
        Shuffle data
        """
        random.shuffle(self.data)

    def load(self) -> Iterator[List]:
        """
        Load data batches as iterator
        """
        while True:
            if self.random:
                self._randomize()
            for batch in self._batchify():
                yield batch

            if not self.infinite:
                break


if __name__ == "__main__":
    dl = DataLoader(np.array(range(10)), infinite=False)
    for d in dl.load():
        print(d)
