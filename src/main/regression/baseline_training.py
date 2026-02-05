from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from main.utils.utils import TrainConfig


def load_feature_store(npy_path: Path) -> pd.DataFrame:
    feats = np.load(npy_path)
    paths_txt = npy_path.with_suffix(".paths.txt")
    keys = paths_txt.read_text().splitlines()

    df = pd.DataFrame(feats)
    df.insert(0, "image_path", keys)
    return df


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
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
        Xtr = Xtr.drop(columns=["image_path", 'State', 'Species'], errors="ignore")
        Xva = Xva.drop(columns=["image_path", 'State', 'Species'], errors="ignore")
        
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
    X = X.drop(columns=["image_path"], errors="ignore")
    return model_wrapper_creator(train_cfg, X).fit(X, y)


def predict(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    X = X.drop(columns=["image_path"], errors="ignore")
    return np.asarray(pipe.predict(X))
