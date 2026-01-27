from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from main.preprocessing.scaling import apply_scaling_train_test
from main.utils.utils import DataType
from main.preprocessing.preproccesing import (
    DatasetPaths,
    read_csv,
    pivot_train_long_to_wide,
    make_features_train,
    make_features_test,
    TARGETS,
)
from main.vision.resnet import ResnetConfig, extract_features
from main.regression.baseline_training import (
    RFConfig,
    load_feature_store,
    merge_features,
    fit_full,
    predict,
)
from main.preprocessing.pca import PCAConfig, fit_pca, transform_pca


# helper functnss
def dedupe_test(df_test: pd.DataFrame) -> pd.DataFrame:
    if "image_path" not in df_test.columns:
        raise ValueError("test.csv must contain image_path")
    return df_test.drop_duplicates(subset=["image_path"]).reset_index(drop=True)


@dataclass(frozen=True)
class PredictConfig:
    pca_components: int = 128
    resnet: ResnetConfig = ResnetConfig(batch_size=32, device="auto")
    rf: RFConfig = RFConfig(n_estimators=1200, min_samples_leaf=2, random_state=42)


def _embedding_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if isinstance(c, str) and c.isdigit()]


def _ensure_features(
    df: pd.DataFrame,
    images_root: Path,
    out_npy: Path,
    resnet_cfg: ResnetConfig,
    image_col: str = "image_path",
    mode: DataType = DataType.VAL,
) -> None:
    paths_txt = out_npy.with_suffix(".paths.txt")
    if out_npy.exists() and paths_txt.exists():
        return
    extract_features(
        df,
        images_root=images_root,
        out_npy=out_npy,
        cfg=resnet_cfg,
        image_col=image_col,
        mode=mode,
    )


def apply_pca_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    emb_cols = _embedding_cols(X_train)

    Ztr = X_train[emb_cols].to_numpy(dtype=np.float32)
    Zte = X_test[emb_cols].to_numpy(dtype=np.float32)

    pca = fit_pca(Ztr, PCAConfig(n_components=n_components))
    Ztr_p = transform_pca(pca, Ztr)
    Zte_p = transform_pca(pca, Zte)

    Xtr2 = X_train.drop(columns=emb_cols).copy()
    Xte2 = X_test.drop(columns=emb_cols).copy()

    cols = [f"pca_{i}" for i in range(n_components)]
    Xtr_p = pd.DataFrame(Ztr_p, columns=cols, index=X_train.index)
    Xte_p = pd.DataFrame(Zte_p, columns=cols, index=X_test.index)

    Xtr_final = pd.concat([Xtr2, Xtr_p], axis=1)
    Xte_final = pd.concat([Xte2, Xte_p], axis=1)

    return Xtr_final, Xte_final


def wide_to_long_predictions(
    image_paths: np.ndarray, preds: np.ndarray
) -> pd.DataFrame:
    out = []
    for i, img in enumerate(image_paths):
        img_id = Path(img).stem
        for j, t in enumerate(TARGETS):
            sample_id = f"{img_id}__{t}"
            out.append({"sample_id": sample_id, "pred": float(preds[i, j])})
    return pd.DataFrame(out)


def main():
    cfg = PredictConfig()
    paths = DatasetPaths()
    model_dir = Path(__file__).parent / "model_data"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_long = read_csv(paths.train_csv)
    train_wide = pivot_train_long_to_wide(train_long)

    X_meta, y, groups, meta = make_features_train(train_wide)
    X_meta = pd.concat([meta, X_meta], axis=1)

    test_df_raw = read_csv(paths.test_csv)
    test_df = dedupe_test(test_df_raw)

    Xt_meta, _, mt = make_features_test(test_df)
    Xt_meta = pd.concat([mt, Xt_meta], axis=1)

    images_root = paths.root

    f_train = model_dir / "features_train.npy"
    f_test = model_dir / "features_test.npy"

    _ensure_features(
        train_long,
        images_root=images_root,
        out_npy=f_train,
        resnet_cfg=cfg.resnet,
        image_col="image_path",
        mode=DataType.TRAIN,
    )
    _ensure_features(
        test_df,
        images_root=images_root,
        out_npy=f_test,
        resnet_cfg=cfg.resnet,
        image_col="image_path",
        mode=DataType.VAL,
    )

    feat_train = load_feature_store(f_train)
    feat_test = load_feature_store(f_test)

    X_train = merge_features(X_meta, feat_train)
    X_test = merge_features(Xt_meta, feat_test)

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    X_train, X_test = apply_scaling_train_test(X_train, X_test)
    X_train, X_test = apply_pca_train_test(
        X_train, X_test, n_components=cfg.pca_components
    )

    pipe = fit_full(cfg.rf, X_train, y)
    preds = predict(pipe, X_test)

    long_pred = wide_to_long_predictions(X_test["image_path"].to_numpy(), preds)

    sample_sub = read_csv(paths.root / "sample_submission.csv")
    sub = sample_sub.merge(
        long_pred,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if sub["pred"].isna().any():
        missing = int(sub["pred"].isna().sum())
        raise ValueError(f"Submission has {missing} missing predictions after merge.")

    out_path = paths.root / "submission.csv"
    sub = sub.drop(columns=["target"]).rename(columns={"pred": "target"})
    sub.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(sub.head())


if __name__ == "__main__":
    main()
