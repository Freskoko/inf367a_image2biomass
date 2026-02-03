from enum import Enum, auto
from pathlib import Path

from attr import dataclass
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


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
    pca_n_components: int = 64

    # resources
    lower_resources = True
    max_cv_groups = 300

    # targets
    TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]
    MANUAL_TARGETS = ["Dry_Total_g", "GDM_g"]

    # todo make this search for hyperparams instead
    def get_model(self):
        return ExtraTreesRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        n_jobs=-1,
        random_state=self.random_state,
    )

    def get_model_grids(self) -> list[dict]:
        et = ExtraTreesRegressor(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        rf = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        svr = SVR()
        knn = KNeighborsRegressor(n_jobs=self.n_jobs)
        gbr = GradientBoostingRegressor(random_state=self.random_state)
        hgb = HistGradientBoostingRegressor(random_state=self.random_state)
        ridge = Ridge(random_state=self.random_state)
        enet = ElasticNet(random_state=self.random_state, max_iter=10000)

        if self.lower_resources:
            return [
                {
                    "model": [et],
                    "model__n_estimators": [400, 800],
                    "model__max_depth": [None, 30],
                    "model__min_samples_leaf": [1, 2],
                    "model__min_samples_split": [2, 5],
                    "model__max_features": ["sqrt", 0.5],
                    "model__bootstrap": [False],
                },
                {
                    "model": [rf],
                    "model__n_estimators": [300, 600],
                    "model__max_depth": [None, 30],
                    "model__min_samples_leaf": [1, 2],
                    "model__min_samples_split": [2, 5],
                    "model__max_features": ["sqrt", 0.5],
                    "model__bootstrap": [True, False],
                },
                {
                    "model": [hgb],
                    "model__max_depth": [None, 6],
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_iter": [200, 400],
                    "model__max_leaf_nodes": [31, 63],
                },
                {
                    "model": [gbr],
                    "model__n_estimators": [200, 400],
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [2, 3],
                    "model__subsample": [0.8, 1.0],
                },
                {
                    "model": [svr],
                    "model__C": [1.0, 10.0, 100.0],
                    "model__gamma": ["scale", "auto"],
                    "model__epsilon": [0.05, 0.1],
                },
                {
                    "model": [knn],
                    "model__n_neighbors": [3, 5, 9, 15],
                    "model__weights": ["uniform", "distance"],
                    "model__p": [1, 2],
                },
                {
                    "model": [ridge],
                    "model__alpha": [0.1, 1.0, 10.0],
                },
                {
                    "model": [enet],
                    "model__alpha": [0.01, 0.1, 1.0],
                    "model__l1_ratio": [0.2, 0.5, 0.8],
                },
            ]

        return [
            {
                "model": [et],
                "model__n_estimators": [400, 800, 1200],
                "model__max_depth": [None, 20, 40],
                "model__min_samples_leaf": [1, 2, 4],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", 0.5, 1.0],
                "model__bootstrap": [False],
            },
            {
                "model": [rf],
                "model__n_estimators": [400, 800, 1200],
                "model__max_depth": [None, 20, 40],
                "model__min_samples_leaf": [1, 2, 4],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", 0.5, 1.0],
                "model__bootstrap": [True, False],
            },
            {
                "model": [hgb],
                "model__max_depth": [None, 6],
                "model__learning_rate": [0.03, 0.1],
                "model__max_iter": [200, 400, 800],
                "model__max_leaf_nodes": [31, 63],
                "model__l2_regularization": [0.0, 0.1],
            },
            {
                "model": [gbr],
                "model__n_estimators": [200, 400, 800],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.9, 1.0],
            },
            {
                "model": [svr],
                "model__C": [1.0, 10.0, 100.0, 300.0],
                "model__gamma": ["scale", "auto"],
                "model__epsilon": [0.01, 0.05, 0.1, 0.2],
            },
            {
                "model": [knn],
                "model__n_neighbors": [3, 5, 9, 15, 25],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
                "model__leaf_size": [20, 40],
            },
            {
                "model": [ridge],
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
            {
                "model": [enet],
                "model__alpha": [0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        ]


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
