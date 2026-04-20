from __future__ import annotations
import argparse
from pathlib import Path
from loguru import logger

from main.preprocessing.pca import apply_pca_train_test
from main.preprocessing.scaling import apply_scaling_train_test
from main.utils.save_file import save_predictions
from main.utils.utils import DatasetPaths, ModelType, TrainConfig, VisionModelConfig
from main.wrangling.combined_data import merge_features
from main.wrangling.img_data import extract_vision_data
from main.wrangling.tabular_data import load_data, wide_to_long_predictions
from main.regression.baseline_training import (
    fit_full,
    load_feature_store,
    predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on full training data and predict on test set."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["tabpfn", "extra_trees"],
        default="tabpfn",
        help="Regression model to use (default: tabpfn).",
    )
    parser.add_argument(
        "--vision-backbone",
        type=str,
        choices=["dino", "resnet"],
        default="dino",
        help="Vision backbone to use for feature extraction (default: dino).",
    )
    return parser.parse_args()


def get_vision_feature_paths(path_cfg: DatasetPaths, backbone: str) -> tuple[Path, Path]:
    # Must match the save locations used in wrangling/img_data.py::extract_vision_data.
    if backbone == "dino":
        return (
            path_cfg.model_dir / "feature_train_dino.npy",
            path_cfg.model_dir / "feature_test_dino.npy",
        )
    if backbone == "resnet":
        return (
            path_cfg.model_dir / "feature_train_resnet.npy",
            path_cfg.model_dir / "feature_test_resnet.npy",
        )
    raise ValueError(f"Unknown backbone: {backbone}")


def main():
    args = parse_args()
    path_cfg = DatasetPaths()
    train_cfg = TrainConfig(model_type=ModelType.from_string(args.model))
    vision_cfg = VisionModelConfig()
    logger.info(
        f"Configs loaded (model={args.model}, vision_backbone={args.vision_backbone})"
    )

    train_wide, test_df, Xtr_meta, Xte_meta, y = load_data(
        path_cfg=path_cfg, train_cfg=train_cfg
    )
    logger.info("Tabular data loaded")

    train_feat_path, test_feat_path = get_vision_feature_paths(
        path_cfg, args.vision_backbone
    )

    if (
        not train_feat_path.exists()
        or not train_feat_path.with_suffix(".paths.txt").exists()
        or not test_feat_path.exists()
        or not test_feat_path.with_suffix(".paths.txt").exists()
    ):
        extract_vision_data(
            path_cfg=path_cfg,
            vision_cfg=vision_cfg,
            train_df=train_wide,
            test_df=test_df,
            backbone=args.vision_backbone,
        )

    img_feat_train = load_feature_store(train_feat_path)
    img_feat_test = load_feature_store(test_feat_path)
    logger.info(f"Vision features ready ({args.vision_backbone})")

    X_vision_train, X_vision_test = apply_pca_train_test(
        img_feat_train, img_feat_test, train_cfg=train_cfg
    )

    X_train = merge_features(Xtr_meta, X_vision_train)
    X_test = merge_features(Xte_meta, X_vision_test)

    # Keep only columns present in both splits so the scaler sees matching columns.
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    X_train, X_test = apply_scaling_train_test(X_train, X_test)
    logger.info("Features merged and scaled")

    pipe = fit_full(train_cfg, X_train, y)
    test_preds = predict(pipe, X_test)
    long_pred = wide_to_long_predictions(
        image_paths=X_test["image_path"], preds=test_preds, train_cfg=train_cfg
    )
    save_predictions(path_cfg, long_pred)
    logger.info("submission.csv written")


if __name__ == "__main__":
    main()