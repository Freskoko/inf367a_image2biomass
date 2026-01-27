from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from main.preprocessing.preproccesing import (
    DatasetPaths,
    read_csv,
    pivot_train_long_to_wide,
    make_features_train_with_id,
)
from main.regression.baselines import build_baseline_models
from main.regression.rf_regressor import (
    load_feature_store,
    merge_features,
    cv_mean_r2,
)
from main.preprocessing.pca import PCAConfig, fit_pca, transform_pca
from sklearn.preprocessing import MinMaxScaler

from main.utils.utils import TrainConfig


def _embedding_cols(X: pd.DataFrame) -> list[str]:
    cols = []
    for c in X.columns:
        if c.isdigit():
            cols.append(c)
    return cols


def apply_pca_to_embeddings(X: pd.DataFrame, n_components: int = 128) -> pd.DataFrame:
    emb_cols = _embedding_cols(X)
    Z = X[emb_cols].to_numpy(dtype=np.float32)

    pca = fit_pca(Z, PCAConfig(n_components=n_components))
    Zp = transform_pca(pca, Z)

    X2 = X.drop(columns=emb_cols).copy()
    pca_cols = [f"pca_{i}" for i in range(Zp.shape[1])]
    Xp = pd.DataFrame(Zp, columns=pca_cols, index=X.index)

    return pd.concat([X2, Xp], axis=1)


def apply_scaler(X: pd.DataFrame) -> pd.DataFrame:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    scaler = MinMaxScaler()
    Z = scaler.fit_transform(X[num_cols])
    X_scaled = X.copy()
    X_scaled[num_cols] = Z
    return X_scaled


def main():
    paths = DatasetPaths()

    df_long = read_csv(paths.train_csv)
    df_wide = pivot_train_long_to_wide(df_long)

    X_meta, y, groups = make_features_train_with_id(df_wide)

    feature_df = load_feature_store(
        Path(__file__).parent / "model_data" / "features_train.npy"
    )
    X = merge_features(X_meta, feature_df)

    X = apply_scaler(X)
    X = apply_pca_to_embeddings(X, n_components=128)

    models = build_baseline_models(random_state=42)

    outputs = {}
    cfg = TrainConfig(n_splits=5, random_state=42)
    for model in models:
        model_name = model.__class__.__name__
        print(f"Training {model_name}")
        outputs[model_name] = cv_mean_r2(cfg, model, X, y, groups)

    Path("model_scores.json").write_text(json.dumps(outputs, default=str))
    for name, score in outputs.items():
        print(f"\n\nMODEL NAME : {name}")
        print("CV mean R2:", score["mean_r2"])
        print("Per-target R2:", score["per_target_r2"])



if __name__ == "__main__":
    main()
