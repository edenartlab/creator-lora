import random

class ConcatDataset:
    def __init__(self, datasets: list):
        self.datasets = datasets
        self.lengths = [
            len(d) for d in datasets
        ]

        self.indices = list(range(sum(self.lengths))))

    def shuffle(self, seed = 0):
        random.seed(seed)
        random.shuffle(self.indices)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx: int):
        idx = self.indices[idx]
        
        assert idx < self.__len__(), f"Invalid idx: {idx}"
        for i, length in enumerate(self.lengths):
            if idx < length:
                return self.datasets[i][idx]
            idx -= length
        raise IndexError("Index out of range")