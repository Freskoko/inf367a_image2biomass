from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from main.utils.utils import ModelType, TrainConfig


def load_feature_store(npy_path: Path) -> pd.DataFrame:
    """Load cached vision features (.npy) plus their sibling .paths.txt key
    file and return a DataFrame whose first column is ``image_path`` and the
    rest are the raw feature dimensions."""
    feats = np.load(npy_path)
    paths_txt = npy_path.with_suffix(".paths.txt")
    keys = paths_txt.read_text().splitlines()

    df = pd.DataFrame(feats)
    df.insert(0, "image_path", keys)
    return df


def _build_preprocessor(X: pd.DataFrame, train_cfg: TrainConfig) -> ColumnTransformer:
    """Build the ColumnTransformer used inside the sklearn ``Pipeline``.

    Numeric columns (the vision features) flow through a nested pipeline
    ``StandardScaler -> PCA(n_components=pca_n_components)`` so that scaler
    statistics and the PCA basis are refit on the training portion of each
    CV fold — this is what prevents validation-fold leakage. Categorical
    columns (if any) get one-hot encoded; the current pipeline has none
    because ``test.csv`` doesn't ship the tabular fields.
    """
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    n_components = min(train_cfg.pca_n_components, len(num_cols))
    num_pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=train_cfg.random_state)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )


def model_wrapper_creator(train_cfg: TrainConfig, X_example):
    """Return the full sklearn ``Pipeline`` = preprocessor + MultiOutputRegressor.

    ``n_jobs`` is forced to 1 when ``train_cfg.model_type == ModelType.TABPFN``
    because (a) TabPFN holds state that doesn't survive worker-process
    serialization and (b) on macOS spawned workers don't inherit the
    ``TABPFN_TOKEN`` env var. TabPFN has internal parallelism so the
    wall-clock cost of dropping external parallelism is small.
    """
    pre = _build_preprocessor(X_example, train_cfg)
    n_jobs = 1 if train_cfg.model_type == ModelType.TABPFN else train_cfg.n_jobs
    model = MultiOutputRegressor(train_cfg.get_model(), n_jobs=n_jobs)
    return Pipeline([("pre", pre), ("model", model)])


TARGET_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}

def weighted_r2_global(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, target_cols: list[str]) -> float:
    """Global weighted R^2 using the competition's target weights.

    All target columns are flattened into a single vector of residuals, each
    row tagged with that target's weight from ``TARGET_WEIGHTS``, and one R^2
    is computed over the pooled vector. Complements the per-target weighted R^2
    (``sum(w_t * R^2_t)``) — we report both because it isn't obvious from the
    task description which formula Kaggle actually scores.
    """
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
    """Run 5-fold GroupKFold CV (grouped by ``image_path``) and return three
    metrics computed on the pooled out-of-fold predictions:

    - ``global_weighted_r2``: per-row-weighted R^2 over the flattened target vector
    - ``per_target_weighted_r2``: ``sum(w_t * R^2_t)`` over per-target R^2 scores
    - ``per_target_r2``: R^2 per target column (dict)

    The composite targets (``GDM_g``, ``Dry_Total_g``) are recomputed from the
    three base predictions inside each fold so evaluation is consistent with
    how the submission is built in ``wide_to_long_predictions``.
    """
    gkf = GroupKFold(n_splits=train_cfg.n_splits)
    target_cols = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g", "Dry_Total_g"]

    # Accumulate out-of-fold predictions so the final weighted R^2 is computed
    # once on the pooled predictions instead of averaging per-fold scores.
    oof_true_parts: list[pd.DataFrame] = []
    oof_pred_parts: list[pd.DataFrame] = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        ytr, yva = y.iloc[tr], y.iloc[va]

        Xtr = Xtr.drop(columns=["image_path"], errors="ignore")
        Xva = Xva.drop(columns=["image_path"], errors="ignore")

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

        oof_true_parts.append(y_true[target_cols])
        oof_pred_parts.append(y_pred[target_cols])

        fold_r2 = weighted_r2_global(y_true, y_pred, target_cols)
        fold_per_target = np.array([r2_score(y_true[c], y_pred[c]) for c in target_cols], dtype=float)
        print(f"fold {fold}: global_weighted_r2={fold_r2:.4f} targets={np.round(fold_per_target, 4)}")

    oof_true = pd.concat(oof_true_parts, axis=0)
    oof_pred = pd.concat(oof_pred_parts, axis=0)

    per_target_r2 = {c: float(r2_score(oof_true[c], oof_pred[c])) for c in target_cols}
    global_weighted_r2 = weighted_r2_global(oof_true, oof_pred, target_cols)
    per_target_weighted_r2 = float(
        sum(TARGET_WEIGHTS[c] * per_target_r2[c] for c in target_cols)
    )

    return {
        "global_weighted_r2": global_weighted_r2,
        "per_target_weighted_r2": per_target_weighted_r2,
        "per_target_r2": per_target_r2,
    }


def fit_full(train_cfg: TrainConfig, X: pd.DataFrame, y: pd.DataFrame) -> Pipeline:
    """Fit the full pipeline on the entire training set for the submission
    run. Drops the ``image_path`` grouping column before handing X to the
    sklearn pipeline; the PCA and scaler inside the pipeline fit on this
    (single) training split — no leakage to worry about here because there is
    no test-side validation."""
    X = X.drop(columns=["image_path"], errors="ignore")
    return model_wrapper_creator(train_cfg, X).fit(X, y)


def predict(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Run the fitted pipeline on X and clip predictions to be non-negative
    (biomass in grams can't be negative; regressors occasionally produce
    slightly negative values near zero-biomass rows)."""
    X = X.drop(columns=["image_path"], errors="ignore")
    preds = np.asarray(pipe.predict(X))
    return np.clip(preds, 0.0, None)
