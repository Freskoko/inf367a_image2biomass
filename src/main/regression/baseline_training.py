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


TARGET_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}

def weighted_r2_global(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, target_cols: list[str]) -> float:
    # stack to long vectors
    y_true = np.concatenate([y_true_df[c].to_numpy() for c in target_cols], axis=0)
    y_pred = np.concatenate([y_pred_df[c].to_numpy() for c in target_cols], axis=0)

    # per-row weights (repeat weight for each row of that target)
    w = np.concatenate(
        [np.full(len(y_true_df), TARGET_WEIGHTS[c], dtype=float) for c in target_cols],
        axis=0,
    )

    # weighted mean of y_true
    y_bar = np.sum(w * y_true) / np.sum(w)

    # weighted R^2
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - y_bar) ** 2)

    # safe guard (rare)
    if ss_tot == 0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def cv_mean_r2(
    train_cfg: TrainConfig,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray,
) -> dict:
    gkf = GroupKFold(n_splits=train_cfg.n_splits)

    fold_scores: list[float] = []
    per_target_scores: list[np.ndarray] = []

    print("y columns:", y.columns.tolist())
    print("y shape:", y.shape)

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        ytr, yva = y.iloc[tr], y.iloc[va]

        # drop non-features
        Xtr = Xtr.drop(columns=["image_path", "State", "Species"], errors="ignore")
        Xva = Xva.drop(columns=["image_path", "State", "Species"], errors="ignore")

        # --- NEW: scaling + PCA INSIDE CV (fit on train fold only) ---
        # Xtr, scaler = apply_scaling_train(Xtr, return_scaler=True)
        # Xva = apply_scaling_train(Xva, scaler=scaler)
        # Xtr, Xva = apply_pca_train_test(Xtr, Xva, train_cfg=train_cfg)
        # ------------------------------------------------------------

        pipe = model_wrapper_creator(train_cfg, Xtr)
        pipe.fit(Xtr, ytr)

        pred = np.asarray(pipe.predict(Xva))
        y_pred = pd.DataFrame(pred, columns=y.columns, index=yva.index)
        y_true = yva.copy()

        y_true["GDM_g"] = y_true["Dry_Green_g"] + y_true["Dry_Clover_g"]
        y_pred["GDM_g"] = y_pred["Dry_Green_g"] + y_pred["Dry_Clover_g"]

        y_true["Dry_Total_g"] = (
            y_true["Dry_Green_g"] + y_true["Dry_Dead_g"] + y_true["Dry_Clover_g"]
        )
        y_pred["Dry_Total_g"] = (
            y_pred["Dry_Green_g"] + y_pred["Dry_Dead_g"] + y_pred["Dry_Clover_g"]
        )

        target_cols = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g", "Dry_Total_g"]

        target_r2 = np.array([r2_score(y_true[c], y_pred[c]) for c in target_cols], dtype=float)
        # mean_r2 = float(np.mean(target_r2))
        global_weighted_r2 = weighted_r2_global(y_true, y_pred, target_cols)

        fold_scores.append(global_weighted_r2)
        per_target_scores.append(target_r2)

        print(f"fold {fold}: global_weighted_r2={global_weighted_r2:.4f} targets={np.round(target_r2, 4)}")

    per_target_mean = np.mean(np.vstack(per_target_scores), axis=0)
    per_target_r2_dict = dict(zip(target_cols, per_target_mean.tolist()))
    print(per_target_r2_dict)

    return {
        "global_weighted_r2": float(np.mean(fold_scores)),
        "per_target_r2": per_target_r2_dict,
    }


def fit_full(train_cfg: TrainConfig, X: pd.DataFrame, y: pd.DataFrame) -> Pipeline:
    X = X.drop(columns=["image_path"], errors="ignore")
    return model_wrapper_creator(train_cfg, X).fit(X, y)


def predict(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    X = X.drop(columns=["image_path"], errors="ignore")
    return np.asarray(pipe.predict(X))
