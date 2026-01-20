# src/training/trainrf.py
from pathlib import Path
from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


TARGETS = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]


@dataclass(frozen=True)
class RFConfig:
    train_wide_path: str = "../data/interim/train_wide.parquet"
    folds_path: str = "../data/interim/folds.parquet"
    out_dir: str = "../data/interim/rf_models"
    n_estimators: int = 800
    max_depth: int | None = None
    min_samples_leaf: int = 1
    random_state: int = 42
    n_jobs: int = -1


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_rf(cfg: RFConfig) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.train_wide_path)
    folds = pd.read_parquet(cfg.folds_path)
    df = df.merge(folds, on="image_id", how="inner")

    feature_cols_num = ["Pre_GSHH_NDVI", "Height_Ave_cm", "month_sin", "month_cos", "year"]
    feature_cols_cat = ["State", "Species"]

    X = df[feature_cols_num + feature_cols_cat]
    Y = df[TARGETS].values
    fold_ids = df["fold"].values

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ],
        remainder="drop",
    )

    base = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("rf", MultiOutputRegressor(base, n_jobs=cfg.n_jobs)),
        ]
    )

    oof = np.zeros_like(Y, dtype=float)

    for fold in sorted(np.unique(fold_ids)):
        tr = fold_ids != fold
        va = fold_ids == fold

        model_fold = clone(model)
        model_fold.fit(X.iloc[tr], Y[tr])

        pred = model_fold.predict(X.iloc[va])
        oof[va] = pred

        fold_path = out_dir / f"rf_fold{fold}.joblib"
        joblib.dump(model_fold, fold_path)

        rmses = [_rmse(Y[va, i], pred[:, i]) for i in range(len(TARGETS))]
        avg = float(np.mean(rmses))

        print(f"Fold {fold} RMSEs:", {t: round(r, 4) for t, r in zip(TARGETS, rmses)})
        print(f"Fold {fold} avg RMSE: {avg:.4f}\n")

    rmses_all = [_rmse(Y[:, i], oof[:, i]) for i in range(len(TARGETS))]
    avg_all = float(np.mean(rmses_all))
    print("OOF RMSEs:", {t: round(r, 4) for t, r in zip(TARGETS, rmses_all)})
    print(f"OOF avg RMSE: {avg_all:.4f}")

    oof_path = out_dir / "oof_preds.npy"
    np.save(oof_path, oof)

    return out_dir


if __name__ == "__main__":
    train_rf(RFConfig())
