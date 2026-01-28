from pathlib import Path
import numpy as np
import pandas as pd

from main.preprocessing.preproccesing import (
    make_features_test,
    make_features_train,
    pivot_train_long_to_wide,
)
from main.utils.utils import DatasetPaths, TrainConfig


def wide_to_long_predictions(
    image_paths: np.ndarray, preds: np.ndarray, train_cfg: TrainConfig
) -> pd.DataFrame:
    pred_wide = (
        pd.DataFrame(preds, columns=train_cfg.TARGETS, index=image_paths)
        .reset_index()
        .rename(columns={"index": "image_path"})
    )

    # construct manual targets
    pred_wide["GDM_g"] = pred_wide["Dry_Green_g"] + pred_wide["Dry_Clover_g"]
    pred_wide["Dry_Total_g"] = pred_wide["Dry_Dead_g"] + pred_wide["GDM_g"]

    long_pred = pred_wide.melt(
        id_vars=["image_path"],
        value_vars=train_cfg.TARGETS + train_cfg.MANUAL_TARGETS,
        var_name="target_name",
        value_name="pred",
    )

    # rebuild sample_id for submission
    long_pred["sample_id"] = (
        long_pred["image_path"].apply(lambda s: Path(s).stem)
        + "__"
        + long_pred["target_name"]
    )
    return long_pred


def dedupe_test(df_test: pd.DataFrame) -> pd.DataFrame:
    if "image_path" not in df_test.columns:
        raise ValueError("test.csv must contain image_path")
    return df_test.drop_duplicates(subset=["image_path"]).reset_index(drop=True)


def load_data(path_cfg: DatasetPaths, train_cfg: TrainConfig):
    train_long = pd.read_csv(path_cfg.train_csv)
    train_wide = pivot_train_long_to_wide(train_long, targets=train_cfg.TARGETS)

    Xtr_meta, y, groups, meta = make_features_train(
        train_wide, targets=train_cfg.TARGETS
    )
    Xtr_meta = pd.concat([meta, Xtr_meta], axis=1)

    test_df_raw = pd.read_csv(path_cfg.test_csv)
    test_df = dedupe_test(test_df_raw)

    Xte_meta, _, mt = make_features_test(test_df)
    Xte_meta = pd.concat([mt, Xte_meta], axis=1)

    return train_wide, test_df, Xtr_meta, Xte_meta, y
