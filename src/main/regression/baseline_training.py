from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from main.utils.utils import TrainConfig
from loguru import logger


def load_feature_store(npy_path: Path) -> pd.DataFrame:
    feats = np.load(npy_path)
    paths_txt = npy_path.with_suffix(".paths.txt")
    keys = paths_txt.read_text().splitlines()

    df = pd.DataFrame(feats)
    df.insert(0, "image_path", keys)
    return df


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if cat_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, cat_cols))

    return ColumnTransformer(
        transformers,
        remainder="drop",
    )


def model_wrapper_creator(train_cfg: TrainConfig, X_example):
    pre = _build_preprocessor(X_example)
    model = MultiOutputRegressor(train_cfg.get_model(), n_jobs=train_cfg.n_jobs)
    return Pipeline([("pre", pre), ("model", model)])


def cv_mean_r2(
    train_cfg: TrainConfig,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray,
) -> dict:
    gkf = GroupKFold(n_splits=train_cfg.n_splits)

    fold_scores: list[float] = []
    per_target_scores: list[np.ndarray] = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        pipe = model_wrapper_creator(train_cfg, Xtr)
        pipe.fit(Xtr, ytr)

        pred = np.asarray(pipe.predict(Xva))
        target_r2 = np.array(
            [r2_score(yva.iloc[:, j], pred[:, j]) for j in range(y.shape[1])],
            dtype=float,
        )
        mean_r2 = float(np.mean(target_r2))

        fold_scores.append(mean_r2)
        per_target_scores.append(target_r2)

        # debug
        # print(f"fold {fold}: mean_r2={mean_r2:.4f} targets={np.round(target_r2, 4)}")

    per_target_mean = np.mean(np.vstack(per_target_scores), axis=0)
    return {"mean_r2": float(np.mean(fold_scores)), "per_target_r2": per_target_mean}


def fit_full(train_cfg: TrainConfig, X: pd.DataFrame, y: pd.DataFrame) -> Pipeline:
    return model_wrapper_creator(train_cfg, X).fit(X, y)


def predict(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipe.predict(X))


def _build_search_pipeline(
    train_cfg: TrainConfig, X_example: pd.DataFrame
) -> Pipeline:
    pre = _build_preprocessor(X_example)
    return Pipeline([("pre", pre), ("model", train_cfg.get_model())])


def fit_full_per_target(
    train_cfg: TrainConfig,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray | None = None,
) -> dict[str, Pipeline]:
    model_grids = train_cfg.get_model_grids()
    models: dict[str, Pipeline] = {}

    X_search, y_search, groups_search, cv = _prepare_search_inputs(
        train_cfg, X, y, groups
    )

    for target in y.columns:
        logger.info(f"GridSearchCV start for target: {target}")
        search = GridSearchCV(
            _build_search_pipeline(train_cfg, X),
            param_grid=model_grids,
            scoring="r2",
            cv=cv,
            n_jobs=train_cfg.n_jobs,
            refit=True,
        )
        if groups_search is None:
            search.fit(X_search, y_search[target])
        else:
            search.fit(X_search, y_search[target], groups=groups_search)

        best_estimator = search.best_estimator_
        best_estimator.fit(X, y[target])
        models[target] = best_estimator
        logger.info(
            f"GridSearchCV done for target: {target} | best_score={search.best_score_:.4f}"
        )

    return models


def cv_search_per_target(
    train_cfg: TrainConfig,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray | None = None,
) -> dict:
    model_grids = train_cfg.get_model_grids()
    best_scores: list[float] = []
    per_target_scores: list[float] = []
    best_params: dict[str, dict] = {}

    X_search, y_search, groups_search, cv = _prepare_search_inputs(
        train_cfg, X, y, groups
    )

    for target in y.columns:
        logger.info(f"GridSearchCV start for target: {target}")
        search = GridSearchCV(
            _build_search_pipeline(train_cfg, X),
            param_grid=model_grids,
            scoring="r2",
            cv=cv,
            n_jobs=train_cfg.n_jobs,
            refit=True,
        )
        if groups_search is None:
            search.fit(X_search, y_search[target])
        else:
            search.fit(X_search, y_search[target], groups=groups_search)

        best_scores.append(float(search.best_score_))
        per_target_scores.append(float(search.best_score_))
        best_params[target] = dict(search.best_params_)
        logger.info(
            f"GridSearchCV done for target: {target} | best_score={search.best_score_:.4f}"
        )

    return {
        "mean_r2": float(np.mean(best_scores)) if best_scores else float("nan"),
        "per_target_r2": np.array(per_target_scores, dtype=float),
        "best_params": best_params,
    }


def predict_per_target(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    targets: list[str],
) -> np.ndarray:
    preds = [np.asarray(models[t].predict(X)).reshape(-1, 1) for t in targets]
    return np.hstack(preds)


def _prepare_search_inputs(
    train_cfg: TrainConfig,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None, KFold | GroupKFold]:
    if train_cfg.lower_resources and groups is not None:
        rng = np.random.default_rng(train_cfg.random_state)
        unique_groups = np.unique(groups)
        if unique_groups.size > train_cfg.max_cv_groups:
            keep_groups = rng.choice(
                unique_groups, size=train_cfg.max_cv_groups, replace=False
            )
            mask = np.isin(groups, keep_groups)
            X_search = X.loc[mask].reset_index(drop=True)
            y_search = y.loc[mask].reset_index(drop=True)
            groups_search = groups[mask]
        else:
            X_search, y_search, groups_search = X, y, groups
    else:
        X_search, y_search, groups_search = X, y, groups

    if groups_search is None:
        cv = KFold(
            n_splits=train_cfg.n_splits,
            shuffle=True,
            random_state=train_cfg.random_state,
        )
    else:
        unique_groups = np.unique(groups_search)
        n_splits = min(train_cfg.n_splits, unique_groups.size)
        if n_splits < 2:
            raise ValueError(
                "Need at least 2 unique groups for GroupKFold when using groups."
            )
        cv = GroupKFold(n_splits=n_splits)

    return X_search, y_search, groups_search, cv
