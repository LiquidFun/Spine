from dataclasses import dataclass


@dataclass
class Number:
    number: int = 0

    def get(self):
        return self.number

    def set(self, number):
        self.number = number


TRAIN_DATASET_SIZE = Number(0)
