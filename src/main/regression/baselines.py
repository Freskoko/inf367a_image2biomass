from dataclasses import dataclass
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


@dataclass(frozen=True)
class RFConfig:
    n_splits: int = 5
    n_estimators: int = 1200
    min_samples_leaf: int = 2
    max_depth: int | None = None
    n_jobs: int = -1
    random_state: int = 42


def build_baseline_models(random_state):
    rf_config = RFConfig()

    return [
        DummyRegressor(),
        RandomForestRegressor(random_state=random_state, n_estimators=50),
        SVR(),
        KNeighborsRegressor(),
        ExtraTreesRegressor(
            n_estimators=rf_config.n_estimators,
            min_samples_leaf=rf_config.min_samples_leaf,
            max_depth=rf_config.max_depth,
            n_jobs=rf_config.n_jobs,
            random_state=rf_config.random_state,
        ),
    ]
