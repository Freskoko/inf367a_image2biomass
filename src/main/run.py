from __future__ import annotations
from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd
from main.preprocessing.pca import apply_pca_train_test
from main.preprocessing.scaling import apply_scaling_train_test
from main.utils.save_file import save_predictions
from main.utils.utils import DatasetPaths, TrainConfig
from main.vision.resnet import VisionModelConfig
from main.regression.baseline_training import (
    load_feature_store,
    fit_full_per_target,
    predict_per_target,
)
from main.wrangling.combined_data import merge_features
from main.wrangling.img_data import extract_vision_data
from main.wrangling.tabular_data import load_data, wide_to_long_predictions

from loguru import logger

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with",
)


def main():
    # 0. configs
    path_cfg = DatasetPaths()
    train_cfg = TrainConfig()
    vision_cfg = VisionModelConfig()
    logger.info("0. Configs loaded")

    # 1. load data
    train_wide, test_df, Xtr_meta, Xte_meta, y = load_data(
        path_cfg=path_cfg, train_cfg=train_cfg
    )
    logger.info("1. Data loaded")

    # 2. run vision extraction on images (if required)
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
    logger.info("2.1 Vision data created (or existed beforehand)")

    # 2. load vision extraction data from file
    img_feat_train = load_feature_store(path_cfg.vision_feats_train)
    img_feat_test = load_feature_store(path_cfg.vision_feats_test)
    logger.info("2.2 Vision data loaded from file")

    # 3 apply PCA on vision output data
    X_vision_train, X_vision_test = apply_pca_train_test(
        img_feat_train, img_feat_test, train_cfg=train_cfg
    )
    logger.info("3. PCA on vision data complete")

    # 4 combine data
    X_train = merge_features(Xtr_meta, X_vision_train)
    X_test = merge_features(Xte_meta, X_vision_test)
    logger.info("4.1 Vision and tabular data combined")

    # TODO: this seems weird, should they not be the same?
    # aha.... test is not the same
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # 4.1 scale data
    X_train, X_test = apply_scaling_train_test(X_train, X_test)
    logger.info("4.2 Train and test data scaled")

    # 5. Multivariate regression
    start_time = time.time()
    
    # preserve metadata
    test_image_paths = X_test["image_path"].copy()
    groups = X_train["image_path"].to_numpy()

    # drop non-numeric features
    X_train = X_train.drop(columns=["image_path"])
    X_test  = X_test.drop(columns=["image_path"])
    
    models = fit_full_per_target(train_cfg, X_train, y, groups=groups)
    logger.info(
        f"5.1 Model fitted to data, time taken = {round(time.time() - start_time)} seconds"
    )

    test_preds = predict_per_target(models, X_test, train_cfg.TARGETS)
    long_pred = wide_to_long_predictions(
        image_paths=test_image_paths,
        preds=test_preds,
        train_cfg=train_cfg,
    )

    save_predictions(path_cfg, long_pred)
    logger.info("5.2 Test prediction complete, saving to csv")


if __name__ == "__main__":
    main()
