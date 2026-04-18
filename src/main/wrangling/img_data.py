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

    if backbone == "dino":
        train_out = path_cfg.model_dir / "feature_train_dino.npy"
        test_out = path_cfg.model_dir / "feature_test_dino.npy"
    elif backbone == "resnet":
        train_out = path_cfg.model_dir / "feature_train_resnet.npy"
        test_out = path_cfg.model_dir / "feature_test_resnet.npy"
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

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