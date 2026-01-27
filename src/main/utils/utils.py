from enum import Enum, auto

from attr import dataclass


class DataType(Enum):
    TRAIN = auto()
    VAL = auto()


@dataclass(frozen=True)
class TrainConfig:
    n_splits: int = 5
    n_jobs: int = -1
    random_state: int = 42
