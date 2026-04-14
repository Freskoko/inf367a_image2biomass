import os
from enum import Enum, auto
from pathlib import Path
from attr import dataclass
from sklearn.ensemble import ExtraTreesRegressor
from tabpfn import TabPFNRegressor

# TabPFN needs a token for the one time model download + license check.
# This one is read only (inference only), so fine to keep in the repo, dont need to hide behind env vars.
TABPFN_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiYzk1YTBjNGMtNDZjYS00MjhiLTgyNDQtNWRlMWNhNGJkZTdkIiwiZXhwIjoxODA3NjM3MzMyfQ.4V6VuHGT9OEHg1lzLr8lEM411T6IHMCuEg1j1yWfo10"
os.environ["TABPFN_TOKEN"] = TABPFN_TOKEN


class DataType(Enum):
    TRAIN = auto()
    VAL = auto()


class ModelType(Enum):
    EXTRA_TREES = auto()
    TABPFN = auto()

    @classmethod
    def from_string(cls, name: str) -> "ModelType":
        # Used to map the --model CLI arg to the enum value.
        mapping = {"tabpfn": cls.TABPFN, "extra_trees": cls.EXTRA_TREES}
        key = name.lower()
        if key not in mapping:
            raise ValueError(
                f"Unknown model '{name}'. Valid options: {list(mapping.keys())}"
            )
        return mapping[key]


@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42

    # CV
    n_splits: int = 5
    n_jobs: int = -1

    # models
    pca_n_components: int = 128
    model_type: ModelType = ModelType.EXTRA_TREES

    # resources
    lower_resources = True
    max_cv_groups = 160

    # targets
    TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]
    MANUAL_TARGETS = ["Dry_Total_g", "GDM_g"]

    def get_model(self):
        if self.model_type == ModelType.TABPFN:
            return TabPFNRegressor(random_state=self.random_state)

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
