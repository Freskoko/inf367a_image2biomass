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
    img_feat_test = load_feature_store(path_cfg.vision_feats_test)
    logger.info("2.2 Vision data loaded from file")

    # 3 apply PCA on vision output data
    X_vision_train, _ = apply_pca_train_test(
        img_feat_train, img_feat_test, train_cfg=train_cfg
    )
    logger.info("3. PCA on vision data complete")

    # 4 combine data
    X_train = merge_features(Xtr_meta, X_vision_train)
    logger.info("4.1 Vision and tabular data combined")

    logger.info("Dropped test data, only doing train")
    X_train = apply_scaling_train(X_train)

    # 4.1 scale data
    logger.info("4.2 Train scaled")

    logger.info("6. Calculating R2 CV score")

    groups = X_train["image_path"].to_numpy()

    X_train = X_train.drop(columns=["image_path"])

    if train_cfg.lower_resources:
        rng = np.random.default_rng(train_cfg.random_state)
        keep_groups = rng.choice(
            np.unique(groups),
            size=train_cfg.max_cv_groups,
            replace=False,
        )

        mask = np.isin(groups, keep_groups)

        X_train_cv = X_train.loc[mask].reset_index(drop=True)
        y_cv = y.loc[mask].reset_index(drop=True)
        groups_cv = groups[mask]

        train_r2_score = cv_mean_r2(
            train_cfg=train_cfg,
            X=X_train_cv,
            y=y_cv,
            groups=groups_cv,
        )
    else:
        train_r2_score = cv_mean_r2(
            train_cfg=train_cfg,
            X=X_train,
            y=y,
            groups=groups,
        )


    print("R2 Score on training data:")
    print("CV mean R2:", train_r2_score["mean_r2"])
    print("Per-target R2:", train_r2_score["per_target_r2"])
    logger.info("End of file")


if __name__ == "__main__":
    main()
