from __future__ import annotations
import argparse

import numpy as np
from loguru import logger
from main.preprocessing.pca import apply_pca_train_test
from main.preprocessing.scaling import apply_scaling_train
from main.utils.utils import DatasetPaths, ModelType, TrainConfig, VisionModelConfig
from main.wrangling.combined_data import merge_features
from main.wrangling.img_data import extract_vision_data
from main.wrangling.tabular_data import load_data
from main.regression.baseline_training import (
    cv_mean_r2,
    load_feature_store,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5-fold GroupKFold CV evaluation.")
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
        choices=["resnet", "dino", "convnext"],
        default="dino",
        help="Vision backbone to use for feature extraction.",
    )
    return parser.parse_args()


# CV-only entry point: fits and evaluates on the train split, ignores test data.
def main():
    args = parse_args()
    path_cfg = DatasetPaths()
    train_cfg = TrainConfig(model_type=ModelType.from_string(args.model))
    vision_cfg = VisionModelConfig(vision_backbone=args.vision_backbone)
    logger.info(
        f"Configs loaded (model={args.model}, vision_backbone={args.vision_backbone})"
    )

    train_wide, test_df, Xtr_meta, _, y = load_data(
        path_cfg=path_cfg, train_cfg=train_cfg
    )
    logger.info("Tabular data loaded")

    # Run the vision model feature extractor only if we don't already have cached features.
    train_feat_path = path_cfg.vision_feats_train_path(args.vision_backbone)
    test_feat_path = path_cfg.vision_feats_test_path(args.vision_backbone)
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

    X_vision_train, _ = apply_pca_train_test(
        img_feat_train, img_feat_test, train_cfg=train_cfg
    )
    X_train = merge_features(Xtr_meta, X_vision_train)
    X_train = apply_scaling_train(X_train)
    logger.info("Features merged and scaled")
    
    logger.info("Device setup: {}".format(train_cfg.device))

    # When lower_resources is on, CV runs on a random subset of image groups
    #to keep TabPFN runs reasonable in wall clock time.
    if train_cfg.lower_resources:
        rng = np.random.default_rng(train_cfg.random_state)
        keep_groups = rng.choice(
            X_train["image_path"].unique(), size=train_cfg.max_cv_groups, replace=False
        )
        mask = X_train["image_path"].isin(keep_groups)
        X_train_cv = X_train.loc[mask].reset_index(drop=True)
        y_cv = y.loc[mask].reset_index(drop=True)
        groups = X_train_cv["image_path"].to_numpy()
        train_r2_score = cv_mean_r2(
            train_cfg=train_cfg, X=X_train_cv, y=y_cv, groups=groups
        )
    else:
        groups = X_train["image_path"].to_numpy()
        train_r2_score = cv_mean_r2(train_cfg=train_cfg, X=X_train, y=y, groups=groups)

    print("CV weighted R2:", train_r2_score["global_weighted_r2"])
    print("Per-target R2:", train_r2_score["per_target_r2"])


if __name__ == "__main__":
    main()
