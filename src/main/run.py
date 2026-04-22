from __future__ import annotations
import argparse
from pathlib import Path
from loguru import logger

from main.utils.save_file import save_predictions
from main.utils.utils import DatasetPaths, ModelType, TrainConfig, VisionModelConfig
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
        choices=["dino", "resnet", "convnext"],
        default="dino",
        help="Vision backbone to use for feature extraction (default: dino).",
    )
    return parser.parse_args()


def get_vision_feature_paths(
    path_cfg: DatasetPaths, backbone: str, image_size: int
) -> tuple[Path, Path]:
    return (
        path_cfg.vision_feats_train_path(backbone, image_size=image_size),
        path_cfg.vision_feats_test_path(backbone, image_size=image_size),
    )


def main():
    args = parse_args()
    path_cfg = DatasetPaths()
    train_cfg = TrainConfig(model_type=ModelType.from_string(args.model))
    vision_cfg = VisionModelConfig(vision_backbone=args.vision_backbone)
    logger.info(
        f"Configs loaded (model={args.model}, vision_backbone={args.vision_backbone})"
    )

    # Load tabular data only for the image-path list. The tabular columns
    # themselves (State, Species, NDVI, height, date) aren't present in
    # test.csv, so the production pipeline is vision-features-only and CV
    # is kept symmetric.
    train_wide, test_df, _, _, y = load_data(path_cfg=path_cfg, train_cfg=train_cfg)
    logger.info("Tabular data loaded")

    # Run the vision model feature extractor only if we don't already have cached features.
    train_feat_path, test_feat_path = get_vision_feature_paths(
        path_cfg, args.vision_backbone, image_size=vision_cfg.image_size
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

    X_train = load_feature_store(train_feat_path)
    X_test = load_feature_store(test_feat_path)
    logger.info(f"Vision features ready ({args.vision_backbone})")

    logger.info("Device setup: {}".format(train_cfg.device))

    pipe = fit_full(train_cfg=train_cfg, X=X_train, y=y)
    preds = predict(pipe=pipe, X=X_test)
    long_pred = wide_to_long_predictions(
        image_paths=X_test["image_path"].to_numpy(),
        preds=preds,
        train_cfg=train_cfg,
    )
    save_predictions(path_cfg=path_cfg, long_pred=long_pred)
    logger.info("submission.csv written")


if __name__ == "__main__":
    main()
