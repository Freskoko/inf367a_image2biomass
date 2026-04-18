from __future__ import annotations

import numpy as np
from main.preprocessing.pca import apply_pca_train_test
from main.preprocessing.scaling import apply_scaling_train
from main.utils.utils import DatasetPaths, TrainConfig
from main.vision.resnet import VisionModelConfig
from main.regression.baseline_training import (
    cv_mean_r2,
    load_feature_store,
)
from main.wrangling.combined_data import merge_features
from main.wrangling.img_data import extract_vision_data
from main.wrangling.tabular_data import load_data

from loguru import logger


# drops test data
def main():
    # 0. configs
    path_cfg = DatasetPaths()
    train_cfg = TrainConfig()
    vision_cfg = VisionModelConfig()
    logger.info("0. Configs loaded")

    # 1. load data
    train_wide, test_df, Xtr_meta, _, y = load_data(
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
    logger.info("2.2 Vision data loaded from file")

    
    # print(img_feat_train.columns)
    img_feat_train_raw = img_feat_train.copy()

    # (optional) make feature columns strings so sklearn/pandas behave nicely
    feat_cols = [c for c in img_feat_train_raw.columns if c != "image_path"]
    img_feat_train_raw = img_feat_train_raw.rename(columns={c: f"vision_{c}" for c in feat_cols})
    X_train = img_feat_train_raw
 
    print("X train", X_train.shape)
    

    logger.info("6. Calculating R2 CV score")

    if train_cfg.lower_resources:
        rng = np.random.default_rng(train_cfg.random_state)
        keep_groups = rng.choice(
            X_train["image_path"].unique(), size=train_cfg.max_cv_groups, replace=False
        )

        mask = X_train["image_path"].isin(keep_groups)
        X_train_cv = X_train.loc[mask].reset_index(drop=True)
        print(X_train_cv.shape)
        y_cv = y.loc[mask].reset_index(drop=True)

        groups = X_train_cv["image_path"].to_numpy()
        train_r2_score = cv_mean_r2(
            train_cfg=train_cfg, X=X_train_cv, y=y_cv, groups=groups
        )
        print("unique image paths:", X_train["image_path"].nunique())
        print("max_cv_groups:", train_cfg.max_cv_groups)
        print("kept unique paths:", len(keep_groups))
        print("rows kept:", X_train_cv.shape[0])

    else:
        groups = X_train["image_path"].to_numpy()
        train_r2_score = cv_mean_r2(train_cfg=train_cfg, X=X_train, y=y, groups=groups)

    print("R2 Score on training data:")
    print("CV mean R2:", train_r2_score["global_weighted_r2"])
    print("Per-target R2:", train_r2_score["per_target_r2"])
    logger.info("End of file")


if __name__ == "__main__":
    main()
