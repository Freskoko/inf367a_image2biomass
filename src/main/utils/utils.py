from enum import Enum, auto
from pathlib import Path

from attr import dataclass
from sklearn.ensemble import ExtraTreesRegressor


class DataType(Enum):
    TRAIN = auto()
    VAL = auto()


@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42

    # CV
    n_splits: int = 5
    n_jobs: int = -1

    # models
    pca_n_components: int = 128

    # resources
    lower_resources = True
    max_cv_groups = 80

    # targets
    TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]
    MANUAL_TARGETS = ["Dry_Total_g", "GDM_g"]

    # todo make this search for hyperparams instead
    def get_model(self):
        return ExtraTreesRegressor(
            n_estimators=275,
            min_samples_leaf=3,
            max_depth=None,
            n_jobs=-1,
            random_state=self.random_state,
        )


@dataclass(frozen=True)
class VisionModelConfig:
    image_size: int = 350
    batch_size: int = 64
    num_workers: int = 0
    device: str = "auto"

    # imagenet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class DatasetPaths:
    def __init__(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)

    root: Path = Path(__file__).parent.parent.parent / "data"
    model_dir: Path = Path(__file__).parent.parent / "model_data"

    @property
    def train_csv(self) -> Path:
        return self.root / "train.csv"

    @property
    def test_csv(self) -> Path:
        return self.root / "test.csv"

    @property
    def vision_feats_train(self):
        return self.model_dir / "features_train.npy"

    @property
    def vision_feats_test(self):
        return self.model_dir / "features_test.npy"
