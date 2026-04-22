import pandas as pd
from main.utils.utils import DataType, DatasetPaths, VisionModelConfig
from main.vision.dino import ImagePathDataset, extract_features


def extract_vision_data(
    path_cfg: DatasetPaths,
    vision_cfg: VisionModelConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    backbone: str = "dino",
):
    train_imgs = train_df["image_path"].astype(str).unique().tolist()
    test_imgs = test_df["image_path"].astype(str).unique().tolist()

    ds_train = ImagePathDataset(
        path_cfg.root,
        train_imgs,
        vision_cfg=vision_cfg,
        mode=DataType.VAL,
    )
    ds_test = ImagePathDataset(
        path_cfg.root,
        test_imgs,
        vision_cfg=vision_cfg,
        mode=DataType.VAL,
    )

    train_out = path_cfg.vision_feats_train_path(
        backbone, image_size=vision_cfg.image_size
    )
    test_out = path_cfg.vision_feats_test_path(
        backbone, image_size=vision_cfg.image_size
    )

    extract_features(
        out_npy=train_out,
        vision_cfg=vision_cfg,
        ds=ds_train,
        backbone=backbone,
    )

    extract_features(
        out_npy=test_out,
        vision_cfg=vision_cfg,
        ds=ds_test,
        backbone=backbone,
    )
