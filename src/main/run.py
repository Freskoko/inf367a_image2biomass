from __future__ import annotations
import argparse
import time

from main.preprocessing.pca import apply_pca_train_test
from main.preprocessing.scaling import apply_scaling_train_test
from main.utils.save_file import save_predictions
from main.utils.utils import DatasetPaths, ModelType, TrainConfig
from main.vision.resnet import VisionModelConfig
from main.regression.baseline_training import (
    load_feature_store,
    fit_full,
    predict,
)
from main.wrangling.combined_data import merge_features
from main.wrangling.img_data import extract_vision_data
from main.wrangling.tabular_data import load_data, wide_to_long_predictions

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model and predict on test set.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["tabpfn", "extra_trees"],
        default="tabpfn",
        help="Regression model to use (default: tabpfn).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path_cfg = DatasetPaths()
    train_cfg = TrainConfig(model_type=ModelType.from_string(args.model))
    vision_cfg = VisionModelConfig()
    logger.info(f"Configs loaded (model={args.model})")

    train_wide, test_df, Xtr_meta, Xte_meta, y = load_data(
        path_cfg=path_cfg, train_cfg=train_cfg
    )
    logger.info("Tabular data loaded")

    # Run the ResNet feature extractor only if we don't already have cached features.
    if (
        not path_cfg.vision_feats_train.with_suffix(".paths.txt").exists()
        or not path_cfg.vision_feats_test.with_suffix(".paths.txt").exists()
    ):
        extract_vision_data(
            path_cfg=path_cfg,
            vision_cfg=vision_cfg,
            train_df=train_wide,
            test_df=test_df,
        )
    img_feat_train = load_feature_store(path_cfg.vision_feats_train)
    img_feat_test = load_feature_store(path_cfg.vision_feats_test)
    logger.info("Vision features ready")

    X_vision_train, X_vision_test = apply_pca_train_test(
        img_feat_train, img_feat_test, train_cfg=train_cfg
    )
    logger.info("PCA applied")

    X_train = merge_features(Xtr_meta, X_vision_train)
    X_test = merge_features(Xte_meta, X_vision_test)

    # Keep only columns present in both splits so train/test stay aligned.
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    X_train, X_test = apply_scaling_train_test(X_train, X_test)
    logger.info("Features merged and scaled")

    start_time = time.time()
    pipe = fit_full(train_cfg, X_train, y)
    logger.info(f"Model fitted in {round(time.time() - start_time)}s")

    test_preds = predict(pipe, X_test)
    long_pred = wide_to_long_predictions(
        image_paths=X_test["image_path"], preds=test_preds, train_cfg=train_cfg
    )
    save_predictions(path_cfg, long_pred)
    logger.info("Predictions written to submission.csv")


if __name__ == "__main__":
    main()
