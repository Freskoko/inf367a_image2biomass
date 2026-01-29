import pandas as pd
from main.utils.utils import DataType, DatasetPaths, VisionModelConfig
from main.vision.resnet import ImagePathDataset, extract_features


def extract_vision_data(
    path_cfg: DatasetPaths,
    vision_cfg: VisionModelConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    train_imgs = train_df["image_path"].astype(str).unique().tolist()
    ds_train = ImagePathDataset(
        path_cfg.root, train_imgs, vision_cfg=vision_cfg, mode=DataType.TRAIN
    )

    test_imgs = test_df["image_path"].astype(str).unique().tolist()
    ds_test = ImagePathDataset(
        path_cfg.root, test_imgs, vision_cfg=vision_cfg, mode=DataType.VAL
    )

    extract_features(
        out_npy=path_cfg.vision_feats_train, vision_cfg=vision_cfg, ds=ds_train
    )

    extract_features(
        out_npy=path_cfg.vision_feats_test, vision_cfg=vision_cfg, ds=ds_test
    )
